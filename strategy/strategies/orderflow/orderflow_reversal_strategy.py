from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analytics.orderflow import OrderFlowAnalyzer

from ...base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    ConfidenceGrade,
    EntryType,
    ExitType,
    MarketRegime,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    SignalStrength,
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


@dataclass(slots=True)
class OrderflowReversalThresholds:
    """
    Пороги reversal-логіки на рівні strategy layer.

    LONG reversal:
    - price still weak / negative
    - but sell-side flow is exhausting
    - buyers start absorbing / taking control

    SHORT reversal:
    - price still strong / positive
    - but buy-side flow is exhausting
    - sellers start absorbing / taking control
    """

    min_trades_count: int = 10
    min_total_volume: float = 0.0

    min_abs_price_change_pct: float = 0.03

    min_abs_cvd_delta_ratio: float = 0.08
    min_abs_volume_delta_ratio: float = 0.10
    min_abs_cvd_change_pct: float = 0.03

    min_aggressive_buy_ratio_for_long: float = 0.52
    min_aggressive_sell_ratio_for_short: float = 0.52

    min_bullish_imbalance_for_long: float = 0.02
    min_bearish_imbalance_for_short: float = 0.02

    min_score_for_signal: float = 0.48
    min_confidence_for_signal: float = 0.52

    preferred_entry_offset_pct: float = 0.0008
    stop_buffer_pct: float = 0.0032
    fallback_rr_ratio: float = 2.2
    max_expected_holding_seconds: int = 360

    require_orderbook_confirmation: bool = False
    require_aggressive_confirmation: bool = True

    def validate(self) -> None:
        if self.min_trades_count < 1:
            raise ValueError("min_trades_count must be >= 1")
        if self.min_total_volume < 0:
            raise ValueError("min_total_volume must be >= 0")
        if self.min_abs_price_change_pct < 0:
            raise ValueError("min_abs_price_change_pct must be >= 0")
        if self.min_abs_cvd_delta_ratio < 0:
            raise ValueError("min_abs_cvd_delta_ratio must be >= 0")
        if self.min_abs_volume_delta_ratio < 0:
            raise ValueError("min_abs_volume_delta_ratio must be >= 0")
        if self.min_abs_cvd_change_pct < 0:
            raise ValueError("min_abs_cvd_change_pct must be >= 0")
        if not 0 <= self.min_aggressive_buy_ratio_for_long <= 1:
            raise ValueError("min_aggressive_buy_ratio_for_long must be between 0 and 1")
        if not 0 <= self.min_aggressive_sell_ratio_for_short <= 1:
            raise ValueError("min_aggressive_sell_ratio_for_short must be between 0 and 1")
        if self.min_bullish_imbalance_for_long < 0:
            raise ValueError("min_bullish_imbalance_for_long must be >= 0")
        if self.min_bearish_imbalance_for_short < 0:
            raise ValueError("min_bearish_imbalance_for_short must be >= 0")
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


@dataclass(slots=True)
class OrderflowReversalSnapshot:
    symbol: str

    last_price: float | None = None
    price_change_pct: float = 0.0

    trades_count: int = 0
    total_volume: float = 0.0

    cvd_change_pct: float = 0.0
    cvd_slope: float = 0.0
    cvd_delta_ratio: float = 0.0

    volume_delta: float = 0.0
    volume_delta_ratio: float = 0.0
    cumulative_volume_delta: float = 0.0

    aggressive_buy_ratio: float = 0.0
    aggressive_sell_ratio: float = 0.0
    aggressive_burst_score: float = 0.0
    aggressive_large_trade_count: int = 0

    orderbook_imbalance_ratio: float = 0.0

    def has_minimum_data(self) -> bool:
        return self.trades_count > 0

    @property
    def buy_absorption_hint(self) -> bool:
        return (
            self.price_change_pct < 0
            and self.cvd_delta_ratio > 0
            and self.volume_delta_ratio > 0
        )

    @property
    def sell_absorption_hint(self) -> bool:
        return (
            self.price_change_pct > 0
            and self.cvd_delta_ratio < 0
            and self.volume_delta_ratio < 0
        )


class OrderflowReversalStrategy(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Strategy reversal по order flow.

    LONG reversal:
    - ціна ще тиснеться вниз або залишається слабкою
    - але CVD / volume delta вже розвертаються вгору
    - агресивні покупці починають домінувати
    - optional orderbook imbalance переходить на bid-side

    SHORT reversal:
    - ціна ще росте або залишається сильною
    - але CVD / volume delta вже розвертаються вниз
    - агресивні продавці починають домінувати
    - optional orderbook imbalance переходить на ask-side
    """

    STRATEGY_NAME = "orderflow_reversal_strategy"
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
        thresholds: OrderflowReversalThresholds | None = None,
        event_bus: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.orderflow_analyzer = orderflow_analyzer
        self.thresholds = thresholds or OrderflowReversalThresholds()

        self.validate_config()
        self.thresholds.validate()

    @property
    def component_name(self) -> str:
        return self.STRATEGY_NAME

    @property
    def priority(self) -> int:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None:
            return strategy_cfg.priority
        return 100

    @property
    def strategy_definition(self) -> StrategyDefinitionConfig | None:
        return self.config.get_strategy(self.STRATEGY_NAME)

    @property
    def supported_regimes(self) -> set[MarketRegime]:
        return {
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.BREAKOUT,
            MarketRegime.SQUEEZE,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.RANGING,
            MarketRegime.UNKNOWN,
        }

    def is_enabled(self) -> bool:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is None:
            return True
        return strategy_cfg.runtime.enabled

    def required_features(self) -> set[str]:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None and strategy_cfg.required_features:
            return set(strategy_cfg.required_features)
        return set(self.DEFAULT_REQUIRED_FEATURES)

    def can_evaluate(self, context: SignalContext) -> bool:
        self.validate_context(context)

        if not self.is_enabled():
            return False

        strategy_cfg = self.strategy_definition
        runtime_cfg = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

        if runtime_cfg.symbols and context.symbol not in runtime_cfg.symbols:
            return False

        if runtime_cfg.timeframes and context.timeframe not in runtime_cfg.timeframes:
            return False

        regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN
        if runtime_cfg.allowed_regimes:
            if regime not in runtime_cfg.allowed_regimes and MarketRegime.UNKNOWN not in runtime_cfg.allowed_regimes:
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

        side = self._detect_reversal_side(snapshot)
        if side == SignalSide.UNKNOWN:
            evaluation.reasons.append("no_orderflow_reversal_detected")
            return evaluation

        score = self._calculate_score(snapshot, side, context)
        confidence = self._calculate_confidence(snapshot, side, context)
        reasons = self._build_reasons(snapshot, side)
        confirmations = self._build_confirmations(snapshot, side, context)

        evaluation.score = score
        evaluation.confidence = confidence
        evaluation.reasons.extend(reasons)

        min_score = max(self._get_min_score(), self.thresholds.min_score_for_signal)
        min_confidence = max(self._get_min_confidence(), self.thresholds.min_confidence_for_signal)

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

    def _resolve_snapshot(self, context: SignalContext) -> OrderflowReversalSnapshot | None:
        snapshot = self._build_snapshot_from_context(context)
        if snapshot is not None:
            return snapshot

        facade = self.orderflow_analyzer
        if facade is None:
            return None

        try:
            return self._build_snapshot_from_facade(context.symbol)
        except Exception:
            self.log_warning(
                "Failed to build reversal snapshot from orderflow facade",
                symbol=context.symbol,
                strategy=self.STRATEGY_NAME,
            )
            return None

    def _build_snapshot_from_context(self, context: SignalContext) -> OrderflowReversalSnapshot | None:
        symbol = context.symbol

        orderflow = context.orderflow if isinstance(context.orderflow, dict) else {}
        cvd = orderflow.get("cvd", {}) if isinstance(orderflow.get("cvd"), dict) else {}
        volume_delta = orderflow.get("volume_delta", {}) if isinstance(orderflow.get("volume_delta"), dict) else {}
        aggressive = (
            orderflow.get("aggressive_trades", {})
            if isinstance(orderflow.get("aggressive_trades"), dict)
            else {}
        )
        imbalance = (
            orderflow.get("orderbook_imbalance", {})
            if isinstance(orderflow.get("orderbook_imbalance"), dict)
            else {}
        )

        snapshot = OrderflowReversalSnapshot(
            symbol=symbol,
            last_price=self._coalesce_float(
                self._feature_value(context, "orderflow.last_price"),
                self._feature_value(context, "price.last"),
                cvd.get("last_price"),
                volume_delta.get("last_price"),
                context.price.last_price if context.price is not None else None,
                context.price.mid_price if context.price is not None else None,
            ),
            price_change_pct=self._coalesce_float(
                self._feature_value(context, "orderflow.price_change_pct"),
                self._feature_value(context, "orderflow.cvd.price_change_pct"),
                self._feature_value(context, "price.change_pct"),
                cvd.get("price_change_pct"),
                0.0,
            ) or 0.0,
            trades_count=int(self._coalesce_int(
                self._feature_value(context, "orderflow.trades_count"),
                self._feature_value(context, "orderflow.cvd.trades_count"),
                self._feature_value(context, "orderflow.volume_delta.trades_count"),
                cvd.get("trades_count"),
                volume_delta.get("trades_count"),
                aggressive.get("trades_count"),
                0,
            ) or 0),
            total_volume=self._coalesce_float(
                self._feature_value(context, "orderflow.total_volume"),
                self._feature_value(context, "orderflow.cvd.total_volume"),
                self._feature_value(context, "orderflow.volume_delta.total_volume"),
                cvd.get("total_volume"),
                volume_delta.get("total_volume"),
                aggressive.get("total_volume"),
                0.0,
            ) or 0.0,
            cvd_change_pct=self._coalesce_float(
                self._feature_value(context, "orderflow.cvd.change_pct"),
                self._feature_value(context, "orderflow.cvd.cvd_change_pct"),
                cvd.get("cvd_change_pct"),
                cvd.get("change_pct"),
                0.0,
            ) or 0.0,
            cvd_slope=self._coalesce_float(
                self._feature_value(context, "orderflow.cvd.slope"),
                self._feature_value(context, "orderflow.cvd.cvd_slope"),
                cvd.get("cvd_slope"),
                cvd.get("slope"),
                0.0,
            ) or 0.0,
            cvd_delta_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.cvd.delta_ratio"),
                cvd.get("delta_ratio"),
                0.0,
            ) or 0.0,
            volume_delta=self._coalesce_float(
                self._feature_value(context, "orderflow.volume_delta.value"),
                self._feature_value(context, "orderflow.volume_delta.volume_delta"),
                volume_delta.get("volume_delta"),
                volume_delta.get("value"),
                0.0,
            ) or 0.0,
            volume_delta_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.volume_delta.delta_ratio"),
                volume_delta.get("delta_ratio"),
                0.0,
            ) or 0.0,
            cumulative_volume_delta=self._coalesce_float(
                self._feature_value(context, "orderflow.volume_delta.cumulative_delta"),
                self._feature_value(context, "orderflow.volume_delta.cumulative_volume_delta"),
                volume_delta.get("cumulative_volume_delta"),
                volume_delta.get("cumulative_delta"),
                0.0,
            ) or 0.0,
            aggressive_buy_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.aggressive_trades.buy_ratio"),
                aggressive.get("buy_ratio"),
                aggressive.get("aggressive_buy_ratio"),
                0.0,
            ) or 0.0,
            aggressive_sell_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.aggressive_trades.sell_ratio"),
                aggressive.get("sell_ratio"),
                aggressive.get("aggressive_sell_ratio"),
                0.0,
            ) or 0.0,
            aggressive_burst_score=self._coalesce_float(
                self._feature_value(context, "orderflow.aggressive_trades.burst_score"),
                aggressive.get("burst_score"),
                0.0,
            ) or 0.0,
            aggressive_large_trade_count=int(self._coalesce_int(
                self._feature_value(context, "orderflow.aggressive_trades.large_trade_count"),
                self._feature_value(context, "orderflow.aggressive_trades.large_trades_count"),
                aggressive.get("large_trade_count"),
                aggressive.get("large_trades_count"),
                0,
            ) or 0),
            orderbook_imbalance_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.orderbook_imbalance.ratio"),
                self._feature_value(context, "orderflow.orderbook_imbalance.imbalance_ratio"),
                imbalance.get("imbalance_ratio"),
                imbalance.get("ratio"),
                0.0,
            ) or 0.0,
        )

        if snapshot.has_minimum_data():
            return snapshot

        return None

    def _build_snapshot_from_facade(self, symbol: str) -> OrderflowReversalSnapshot:
        facade = self.orderflow_analyzer
        if facade is None:
            raise ValueError("orderflow_analyzer is not configured")

        cvd_stats = self._safe_get_latest_stats(facade, "cvd", symbol)
        vd_stats = self._safe_get_latest_stats(facade, "volume_delta", symbol)
        aggressive_stats = self._safe_get_latest_stats(facade, "aggressive_trades", symbol)
        imbalance_stats = self._safe_get_latest_stats(facade, "orderbook_imbalance", symbol)

        return OrderflowReversalSnapshot(
            symbol=symbol,
            last_price=self._coalesce_float(
                self._read(cvd_stats, "last_price"),
                self._read(vd_stats, "last_price"),
                self._read(aggressive_stats, "last_price"),
                self._read(imbalance_stats, "mid_price"),
            ),
            price_change_pct=self._coalesce_float(
                self._read(cvd_stats, "price_change_pct"),
                self._read(vd_stats, "price_change_pct"),
                0.0,
            ) or 0.0,
            trades_count=int(self._coalesce_int(
                self._read(cvd_stats, "trades_count"),
                self._read(vd_stats, "trades_count"),
                self._read(aggressive_stats, "trades_count"),
                0,
            ) or 0),
            total_volume=self._coalesce_float(
                self._read(cvd_stats, "total_volume"),
                self._read(vd_stats, "total_volume"),
                self._read(aggressive_stats, "total_volume"),
                0.0,
            ) or 0.0,
            cvd_change_pct=self._coalesce_float(
                self._read(cvd_stats, "cvd_change_pct"),
                0.0,
            ) or 0.0,
            cvd_slope=self._coalesce_float(
                self._read(cvd_stats, "cvd_slope"),
                0.0,
            ) or 0.0,
            cvd_delta_ratio=self._coalesce_float(
                self._read(cvd_stats, "delta_ratio"),
                0.0,
            ) or 0.0,
            volume_delta=self._coalesce_float(
                self._read(vd_stats, "volume_delta"),
                0.0,
            ) or 0.0,
            volume_delta_ratio=self._coalesce_float(
                self._read(vd_stats, "delta_ratio"),
                0.0,
            ) or 0.0,
            cumulative_volume_delta=self._coalesce_float(
                self._read(vd_stats, "cumulative_volume_delta"),
                self._read(vd_stats, "cumulative_delta"),
                0.0,
            ) or 0.0,
            aggressive_buy_ratio=self._coalesce_float(
                self._read(aggressive_stats, "buy_ratio"),
                self._read(aggressive_stats, "aggressive_buy_ratio"),
                0.0,
            ) or 0.0,
            aggressive_sell_ratio=self._coalesce_float(
                self._read(aggressive_stats, "sell_ratio"),
                self._read(aggressive_stats, "aggressive_sell_ratio"),
                0.0,
            ) or 0.0,
            aggressive_burst_score=self._coalesce_float(
                self._read(aggressive_stats, "burst_score"),
                0.0,
            ) or 0.0,
            aggressive_large_trade_count=int(self._coalesce_int(
                self._read(aggressive_stats, "large_trade_count"),
                self._read(aggressive_stats, "large_trades_count"),
                0,
            ) or 0),
            orderbook_imbalance_ratio=self._coalesce_float(
                self._read(imbalance_stats, "imbalance_ratio"),
                self._read(imbalance_stats, "ratio"),
                0.0,
            ) or 0.0,
        )

    def _safe_get_latest_stats(self, facade: OrderFlowAnalyzer, module_name: str, symbol: str) -> Any:
        module = getattr(facade, module_name, None)
        if module is None and hasattr(facade, "get_module"):
            module = facade.get_module(module_name)

        if module is None:
            return None

        getter = getattr(module, "get_latest_stats", None)
        if not callable(getter):
            return None

        return getter(symbol)

    # ------------------------------------------------------------------
    # Reversal logic
    # ------------------------------------------------------------------

    def _detect_reversal_side(self, snapshot: OrderflowReversalSnapshot) -> SignalSide:
        long_ok = self._is_long_reversal(snapshot)
        short_ok = self._is_short_reversal(snapshot)

        if long_ok and not short_ok:
            return SignalSide.LONG
        if short_ok and not long_ok:
            return SignalSide.SHORT
        return SignalSide.UNKNOWN

    def _is_long_reversal(self, snapshot: OrderflowReversalSnapshot) -> bool:
        if snapshot.price_change_pct > -self.thresholds.min_abs_price_change_pct:
            return False

        if snapshot.cvd_delta_ratio < self.thresholds.min_abs_cvd_delta_ratio:
            return False

        if snapshot.volume_delta_ratio < self.thresholds.min_abs_volume_delta_ratio:
            return False

        if snapshot.cvd_change_pct < self.thresholds.min_abs_cvd_change_pct:
            return False

        if self.thresholds.require_aggressive_confirmation:
            if snapshot.aggressive_buy_ratio < self.thresholds.min_aggressive_buy_ratio_for_long:
                return False
            if snapshot.aggressive_buy_ratio <= snapshot.aggressive_sell_ratio:
                return False

        if self.thresholds.require_orderbook_confirmation:
            if snapshot.orderbook_imbalance_ratio < self.thresholds.min_bullish_imbalance_for_long:
                return False

        return True

    def _is_short_reversal(self, snapshot: OrderflowReversalSnapshot) -> bool:
        if snapshot.price_change_pct < self.thresholds.min_abs_price_change_pct:
            return False

        if snapshot.cvd_delta_ratio > -self.thresholds.min_abs_cvd_delta_ratio:
            return False

        if snapshot.volume_delta_ratio > -self.thresholds.min_abs_volume_delta_ratio:
            return False

        if snapshot.cvd_change_pct > -self.thresholds.min_abs_cvd_change_pct:
            return False

        if self.thresholds.require_aggressive_confirmation:
            if snapshot.aggressive_sell_ratio < self.thresholds.min_aggressive_sell_ratio_for_short:
                return False
            if snapshot.aggressive_sell_ratio <= snapshot.aggressive_buy_ratio:
                return False

        if self.thresholds.require_orderbook_confirmation:
            if snapshot.orderbook_imbalance_ratio > -self.thresholds.min_bearish_imbalance_for_short:
                return False

        return True

    def _calculate_score(
        self,
        snapshot: OrderflowReversalSnapshot,
        side: SignalSide,
        context: SignalContext,
    ) -> float:
        price_component = self._normalize_percent(abs(snapshot.price_change_pct), scale=1.25)
        cvd_component = self._normalize_percent(abs(snapshot.cvd_change_pct), scale=1.25)
        cvd_ratio_component = self._normalize_ratio(abs(snapshot.cvd_delta_ratio), scale=0.35)
        volume_ratio_component = self._normalize_ratio(abs(snapshot.volume_delta_ratio), scale=0.35)

        aggression_component = (
            snapshot.aggressive_buy_ratio if side == SignalSide.LONG else snapshot.aggressive_sell_ratio
        )
        imbalance_component = self._normalize_ratio(abs(snapshot.orderbook_imbalance_ratio), scale=0.20)
        absorption_component = 1.0 if (
            snapshot.buy_absorption_hint if side == SignalSide.LONG else snapshot.sell_absorption_hint
        ) else 0.0
        trades_component = min(snapshot.trades_count / max(self.thresholds.min_trades_count * 2, 1), 1.0)

        raw_score = (
            (price_component * 0.14)
            + (cvd_component * 0.17)
            + (cvd_ratio_component * 0.18)
            + (volume_ratio_component * 0.18)
            + (aggression_component * 0.14)
            + (imbalance_component * 0.06)
            + (absorption_component * 0.09)
            + (trades_component * 0.04)
        )

        weighted_score = raw_score
        weighted_score *= self._category_weight()
        weighted_score *= self._regime_adjustment(context)
        weighted_score *= self._strategy_weight()

        return max(0.0, weighted_score)

    def _calculate_confidence(
        self,
        snapshot: OrderflowReversalSnapshot,
        side: SignalSide,
        context: SignalContext,
    ) -> float:
        components: list[float] = [
            self._normalize_percent(abs(snapshot.price_change_pct), scale=1.10),
            self._normalize_percent(abs(snapshot.cvd_change_pct), scale=1.10),
            self._normalize_ratio(abs(snapshot.cvd_delta_ratio), scale=0.30),
            self._normalize_ratio(abs(snapshot.volume_delta_ratio), scale=0.30),
            min(snapshot.trades_count / max(self.thresholds.min_trades_count * 2, 1), 1.0),
        ]

        if side == SignalSide.LONG:
            components.append(snapshot.aggressive_buy_ratio)
            components.append(1.0 if snapshot.buy_absorption_hint else 0.35)
        elif side == SignalSide.SHORT:
            components.append(snapshot.aggressive_sell_ratio)
            components.append(1.0 if snapshot.sell_absorption_hint else 0.35)

        if self.thresholds.require_orderbook_confirmation:
            components.append(
                1.0 if abs(snapshot.orderbook_imbalance_ratio) > 0 else 0.25
            )
        else:
            components.append(self._normalize_ratio(abs(snapshot.orderbook_imbalance_ratio), scale=0.20))

        if context.regime is not None and context.regime.regime in self.supported_regimes:
            components.append(0.75)

        if context.price is not None and context.price.spread_bps is not None:
            components.append(1.0 if context.price.spread_bps <= self.config.filters.max_spread_bps else 0.30)

        confidence = sum(components) / len(components) if components else 0.0
        return max(0.0, min(confidence, 1.0))

    def _build_reasons(
        self,
        snapshot: OrderflowReversalSnapshot,
        side: SignalSide,
    ) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.extend(
                [
                    "down_move_shows_orderflow_exhaustion",
                    "bullish_reversal_pressure_detected",
                    "cvd_turns_positive_against_price_weakness",
                    "volume_delta_turns_positive_against_price_weakness",
                ]
            )
            if snapshot.buy_absorption_hint:
                reasons.append("buy_absorption_detected")
            if snapshot.aggressive_buy_ratio > snapshot.aggressive_sell_ratio:
                reasons.append("aggressive_buyers_take_control")
            if snapshot.orderbook_imbalance_ratio > 0:
                reasons.append("orderbook_shifts_to_bid_support")
        elif side == SignalSide.SHORT:
            reasons.extend(
                [
                    "up_move_shows_orderflow_exhaustion",
                    "bearish_reversal_pressure_detected",
                    "cvd_turns_negative_against_price_strength",
                    "volume_delta_turns_negative_against_price_strength",
                ]
            )
            if snapshot.sell_absorption_hint:
                reasons.append("sell_absorption_detected")
            if snapshot.aggressive_sell_ratio > snapshot.aggressive_buy_ratio:
                reasons.append("aggressive_sellers_take_control")
            if snapshot.orderbook_imbalance_ratio < 0:
                reasons.append("orderbook_shifts_to_ask_pressure")

        if snapshot.aggressive_burst_score > 0:
            reasons.append("aggressive_flow_burst_present")

        if snapshot.trades_count >= self.thresholds.min_trades_count:
            reasons.append("sufficient_trade_sample")

        return reasons

    def _build_confirmations(
        self,
        snapshot: OrderflowReversalSnapshot,
        side: SignalSide,
        context: SignalContext,
    ) -> list[str]:
        confirmations: list[str] = []

        if side == SignalSide.LONG:
            if snapshot.cvd_slope > 0:
                confirmations.append("positive_cvd_slope")
            if snapshot.volume_delta > 0:
                confirmations.append("positive_volume_delta")
            if snapshot.orderbook_imbalance_ratio > 0:
                confirmations.append("positive_orderbook_imbalance")
            if snapshot.buy_absorption_hint:
                confirmations.append("buy_absorption_confirmation")
        elif side == SignalSide.SHORT:
            if snapshot.cvd_slope < 0:
                confirmations.append("negative_cvd_slope")
            if snapshot.volume_delta < 0:
                confirmations.append("negative_volume_delta")
            if snapshot.orderbook_imbalance_ratio < 0:
                confirmations.append("negative_orderbook_imbalance")
            if snapshot.sell_absorption_hint:
                confirmations.append("sell_absorption_confirmation")

        if context.price is not None and context.price.spread_bps is not None:
            if context.price.spread_bps <= self.config.filters.max_spread_bps:
                confirmations.append("spread_filter_ok")

        if context.regime is not None:
            if context.regime.regime in self.supported_regimes:
                confirmations.append("regime_alignment_ok")

        return confirmations

    # ------------------------------------------------------------------
    # Signal build
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        *,
        context: SignalContext,
        snapshot: OrderflowReversalSnapshot,
        side: SignalSide,
        score: float,
        confidence: float,
        reasons: list[str],
        confirmations: list[str],
    ) -> StrategySignal:
        entry_plan = self._build_entry_plan(context, snapshot, side)
        exit_plan = self._build_exit_plan(context, snapshot, side, entry_plan)
        invalidation_plan = self._build_invalidation_plan(context, snapshot, side, entry_plan)
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
                "source": "orderflow_reversal_strategy",
                "analytics_fallback_enabled": self.orderflow_analyzer is not None,
                "orderflow_snapshot": {
                    "price_change_pct": snapshot.price_change_pct,
                    "trades_count": snapshot.trades_count,
                    "total_volume": snapshot.total_volume,
                    "cvd_change_pct": snapshot.cvd_change_pct,
                    "cvd_slope": snapshot.cvd_slope,
                    "cvd_delta_ratio": snapshot.cvd_delta_ratio,
                    "volume_delta": snapshot.volume_delta,
                    "volume_delta_ratio": snapshot.volume_delta_ratio,
                    "cumulative_volume_delta": snapshot.cumulative_volume_delta,
                    "aggressive_buy_ratio": snapshot.aggressive_buy_ratio,
                    "aggressive_sell_ratio": snapshot.aggressive_sell_ratio,
                    "aggressive_burst_score": snapshot.aggressive_burst_score,
                    "aggressive_large_trade_count": snapshot.aggressive_large_trade_count,
                    "orderbook_imbalance_ratio": snapshot.orderbook_imbalance_ratio,
                    "buy_absorption_hint": snapshot.buy_absorption_hint,
                    "sell_absorption_hint": snapshot.sell_absorption_hint,
                    "last_price": snapshot.last_price,
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
        snapshot: OrderflowReversalSnapshot,
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
            entry_type=getattr(self.config.builders, "default_entry_type", EntryType.MARKET),
            price=entry_price,
            confirmation_required=False,
            notes=[
                "entry_generated_from_orderflow_reversal",
                "prefer_execution_after_absorption_confirmation",
            ],
            metadata={
                "reference_price": ref_price,
                "entry_offset_pct": self.thresholds.preferred_entry_offset_pct,
            },
        )

    def _build_exit_plan(
        self,
        context: SignalContext,
        snapshot: OrderflowReversalSnapshot,
        side: SignalSide,
        entry_plan: EntryPlan,
    ) -> ExitPlan:
        ref_price = entry_plan.price or self._resolve_reference_price(context, snapshot)

        stop_loss = None
        tp_levels: list[TargetPlan] = []

        if ref_price is not None:
            stop_buffer = ref_price * self.thresholds.stop_buffer_pct
            rr_ratio = getattr(self.config.builders, "default_rr_ratio", self.thresholds.fallback_rr_ratio)
            rr_ratio = rr_ratio if rr_ratio and rr_ratio > 0 else self.thresholds.fallback_rr_ratio

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
            partial_exit_enabled=getattr(self.config.builders, "enable_partial_take_profit", True),
            metadata={
                "strategy": self.STRATEGY_NAME,
                "rr_ratio": getattr(self.config.builders, "default_rr_ratio", self.thresholds.fallback_rr_ratio),
            },
        )

    def _build_invalidation_plan(
        self,
        context: SignalContext,
        snapshot: OrderflowReversalSnapshot,
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
            reason="orderflow_reversal_failed",
            conditions=[
                "cvd_realigns_with_previous_trend",
                "volume_delta_realigns_with_previous_trend",
                "aggressive_flow_returns_to_previous_side",
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
                "generated_from_orderflow_reversal_strategy",
            ],
            metadata={
                "strategy_name": self.STRATEGY_NAME,
                "timeframe": str(context.timeframe),
            },
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_min_confidence(self) -> float:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None:
            return strategy_cfg.runtime.min_confidence
        return self.config.runtime.min_confidence

    def _get_min_score(self) -> float:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None:
            return strategy_cfg.runtime.min_score
        return self.config.runtime.min_score

    def _category_weight(self) -> float:
        try:
            return float(self.config.weighting.category_weights.get(self.CATEGORY, 1.0))
        except Exception:
            return 1.0

    def _strategy_weight(self) -> float:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is None:
            return 1.0
        try:
            return float(strategy_cfg.weight)
        except Exception:
            return 1.0

    def _regime_adjustment(self, context: SignalContext) -> float:
        regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN
        try:
            return float(self.config.weighting.regime_adjustments.get(regime, 1.0))
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _resolve_priority(self, confidence: float) -> SignalPriority:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return SignalPriority.HIGH
        if confidence >= cfg.low_threshold:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    def _map_strength(self, confidence: float) -> SignalStrength:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return SignalStrength.STRONG
        if confidence >= cfg.medium_threshold:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _map_confidence_grade(self, confidence: float) -> ConfidenceGrade:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return ConfidenceGrade.VERY_HIGH
        if confidence >= cfg.medium_threshold:
            return ConfidenceGrade.HIGH
        if confidence >= cfg.low_threshold:
            return ConfidenceGrade.MEDIUM
        if confidence >= cfg.very_low_threshold:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _resolve_reference_price(
        self,
        context: SignalContext,
        snapshot: OrderflowReversalSnapshot,
    ) -> float | None:
        if context.price is not None:
            if context.price.mid_price is not None:
                return context.price.mid_price
            if context.price.last_price is not None:
                return context.price.last_price
        return snapshot.last_price

    def _feature_value(self, context: SignalContext, name: str) -> Any:
        snapshot = context.get_feature_snapshot(name)
        if snapshot is None:
            return None
        return snapshot.value

    @staticmethod
    def _read(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _coalesce_float(*values: Any) -> float | None:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coalesce_int(*values: Any) -> int | None:
        for value in values:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _normalize_percent(value: float, scale: float = 1.0) -> float:
        if scale <= 0:
            scale = 1.0
        return max(0.0, min(abs(value) / scale, 1.0))

    @staticmethod
    def _normalize_ratio(value: float, scale: float = 1.0) -> float:
        if scale <= 0:
            scale = 1.0
        return max(0.0, min(abs(value) / scale, 1.0))