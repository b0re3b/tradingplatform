from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from analytics.orderflow import OrderFlowAnalyzer
from analytics.orderflow.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OrderFlowKey,
    make_orderflow_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    orderflow_key_to_dict,
    orderflow_key_to_string,
)
from core.event_bus import EventBus, EventPriority
from core.logger import get_logger

from ...base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    ConfidenceGrade,
    MarketRegime,
    SignalPriority,
    SignalStrength,
    StrategyCategory,
)
from ...models import SignalContext, StrategyEvaluation


@dataclass(slots=True)
class OrderflowCompositeSnapshot:
    """
    Strategy-level normalized view over analytics.orderflow metrics.

    This is intentionally not an analytics model.
    It is a strategy-side projection that lets concrete orderflow strategies
    consume CVD, volume delta, aggressive trades and orderbook imbalance through
    one stable contract.

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    exchange: str
    market_type: str
    symbol: str
    timeframe: str

    exchange_symbol: str | None = None
    timestamp: float | None = None
    source: str = "unknown"

    last_price: float | None = None
    price_change: float | None = None
    price_change_pct: float = 0.0

    window_seconds: float = 0.0
    trades_count: int = 0
    total_volume: float = 0.0
    total_notional: float = 0.0

    # CVD
    cvd_value: float = 0.0
    cvd_open: float = 0.0
    cvd_high: float = 0.0
    cvd_low: float = 0.0
    cvd_close: float = 0.0
    cvd_change: float = 0.0
    cvd_change_pct: float = 0.0
    cvd_slope: float = 0.0
    cvd_delta_ratio: float = 0.0
    cvd_buy_ratio: float = 0.0
    cvd_sell_ratio: float = 0.0

    # Volume delta
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    volume_delta: float = 0.0
    notional_delta: float = 0.0
    volume_delta_ratio: float = 0.0
    cumulative_volume_delta: float = 0.0
    cumulative_notional_delta: float = 0.0
    volume_buy_ratio: float = 0.0
    volume_sell_ratio: float = 0.0
    avg_trade_size: float = 0.0
    avg_trade_notional: float = 0.0

    # Aggressive trades
    aggressive_buy_count: int = 0
    aggressive_sell_count: int = 0
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    aggressive_buy_notional: float = 0.0
    aggressive_sell_notional: float = 0.0
    aggressive_net_volume_delta: float = 0.0
    aggressive_net_notional_delta: float = 0.0
    aggressive_buy_ratio: float = 0.0
    aggressive_sell_ratio: float = 0.0
    aggressive_burst_score: float = 0.0
    large_buy_trades: int = 0
    large_sell_trades: int = 0
    aggressive_avg_trade_size: float = 0.0
    aggressive_avg_trade_notional: float = 0.0

    # Orderbook imbalance
    orderbook_bid_volume: float = 0.0
    orderbook_ask_volume: float = 0.0
    orderbook_imbalance_ratio: float = 0.0
    orderbook_imbalance_diff: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid_price: float | None = None
    depth_levels_used: int = 0

    raw_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.exchange = normalize_exchange(self.exchange)
        self.market_type = normalize_market_type(self.market_type)
        self.symbol = normalize_symbol(self.symbol)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = str(self.exchange_symbol or self.symbol)

        self.timestamp = _safe_float(self.timestamp)
        self.last_price = _safe_float(self.last_price)
        self.price_change = _safe_float(self.price_change)
        self.best_bid = _safe_float(self.best_bid)
        self.best_ask = _safe_float(self.best_ask)
        self.spread = _safe_float(self.spread)
        self.mid_price = _safe_float(self.mid_price)

        self.trades_count = int(max(self.trades_count, 0))
        self.aggressive_buy_count = int(max(self.aggressive_buy_count, 0))
        self.aggressive_sell_count = int(max(self.aggressive_sell_count, 0))
        self.large_buy_trades = int(max(self.large_buy_trades, 0))
        self.large_sell_trades = int(max(self.large_sell_trades, 0))
        self.depth_levels_used = int(max(self.depth_levels_used, 0))

    @property
    def key(self) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def orderflow_key(self) -> OrderFlowKey:
        return self.key

    @property
    def scope(self) -> dict[str, str]:
        return orderflow_key_to_dict(self.key)

    @property
    def scope_key(self) -> str:
        return orderflow_key_to_string(self.key)

    @property
    def aggressive_large_trade_count(self) -> int:
        """
        Backward-compatible aggregate for old strategy code.

        New directional logic should prefer:
        - large_buy_trades for LONG
        - large_sell_trades for SHORT
        """
        return self.large_buy_trades + self.large_sell_trades

    @property
    def signed_orderbook_imbalance(self) -> float:
        """
        Signed orderbook value for strategy logic.

        Prefer this over raw orderbook_imbalance_ratio because analytics can
        configure ratio as either 0..1 or -1..1, while imbalance_diff is signed.
        """
        return self.orderbook_imbalance_diff

    @property
    def has_orderbook(self) -> bool:
        return (
            self.orderbook_bid_volume > 0.0
            or self.orderbook_ask_volume > 0.0
            or self.depth_levels_used > 0
        )

    @property
    def has_aggressive_flow(self) -> bool:
        return (
            self.aggressive_buy_count > 0
            or self.aggressive_sell_count > 0
            or self.aggressive_buy_volume > 0.0
            or self.aggressive_sell_volume > 0.0
        )

    def has_minimum_data(self) -> bool:
        return self.trades_count > 0 or self.total_volume > 0.0 or self.has_orderbook

    def directional_large_trades(self, side: str) -> int:
        normalized = str(side).strip().lower()
        if normalized in {"long", "buy", "bullish"}:
            return self.large_buy_trades
        if normalized in {"short", "sell", "bearish"}:
            return self.large_sell_trades
        return 0

    def directional_aggressive_ratio(self, side: str) -> float:
        normalized = str(side).strip().lower()
        if normalized in {"long", "buy", "bullish"}:
            return self.aggressive_buy_ratio
        if normalized in {"short", "sell", "bearish"}:
            return self.aggressive_sell_ratio
        return 0.0

    def directional_aggressive_notional_delta(self, side: str) -> float:
        normalized = str(side).strip().lower()
        if normalized in {"long", "buy", "bullish"}:
            return self.aggressive_net_notional_delta
        if normalized in {"short", "sell", "bearish"}:
            return -self.aggressive_net_notional_delta
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = list(self.key)
        payload["scope"] = self.scope
        payload["scope_key"] = self.scope_key
        payload["aggressive_large_trade_count"] = self.aggressive_large_trade_count
        payload["signed_orderbook_imbalance"] = self.signed_orderbook_imbalance
        payload["has_orderbook"] = self.has_orderbook
        payload["has_aggressive_flow"] = self.has_aggressive_flow
        return payload


class OrderflowStrategyBase(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Shared base for strategy/strategies/orderflow.

    Responsibilities:
    - core.logger through get_logger();
    - EventBus evaluation event emission;
    - StrategyConfig/runtime filters;
    - scoped analytics.orderflow access;
    - CVD / volume delta / aggressive trades / orderbook metric resolution;
    - composite orderflow snapshot for concrete strategies;
    - confidence, priority, strength and normalization helpers.

    Concrete strategies should keep only trading logic:
    detection, scoring, confirmations and signal plan construction.
    """

    STRATEGY_NAME: str = "orderflow_strategy"
    CATEGORY: StrategyCategory = StrategyCategory.ORDERFLOW

    DEFAULT_REQUIRED_FEATURES: set[str] = set()
    REQUIRED_FEATURES: set[str] = set()

    DEFAULT_EXCHANGE: str = DEFAULT_EXCHANGE
    DEFAULT_MARKET_TYPE: str = DEFAULT_MARKET_TYPE
    DEFAULT_TIMEFRAME: str = DEFAULT_TIMEFRAME

    # Keep old symbol-only fallback available during migration, but do not
    # make it the primary strategy path.
    ALLOW_LEGACY_SYMBOL_FALLBACK: bool = True

    ORDERFLOW_METRIC_NAMES: tuple[str, ...] = (
        "cvd",
        "volume_delta",
        "aggressive_trades",
        "orderbook_imbalance",
    )

    def __init__(
        self,
        config: StrategyConfig,
        *,
        orderflow_analyzer: OrderFlowAnalyzer | None = None,
        event_bus: EventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        resolved_logger = logger or get_logger(
            __name__,
            event_type="strategy",
            strategies=self.STRATEGY_NAME,
        )
        super().__init__(config=config, event_bus=event_bus, logger=resolved_logger)
        self.orderflow_analyzer = orderflow_analyzer

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

    def is_enabled(self) -> bool:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is None:
            return True
        return strategy_cfg.runtime.enabled

    def required_features(self) -> set[str]:
        strategy_cfg = self.strategy_definition
        if strategy_cfg is not None and strategy_cfg.required_features:
            return set(strategy_cfg.required_features)

        if self.DEFAULT_REQUIRED_FEATURES:
            return set(self.DEFAULT_REQUIRED_FEATURES)

        return set(self.REQUIRED_FEATURES)

    def _runtime_allows_context(self, context: SignalContext) -> bool:
        strategy_cfg = self.strategy_definition
        runtime_cfg = strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

        if runtime_cfg.symbols and context.symbol not in runtime_cfg.symbols:
            return False

        if runtime_cfg.timeframes and context.timeframe not in runtime_cfg.timeframes:
            return False

        regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN

        if runtime_cfg.allowed_regimes:
            if (
                regime not in runtime_cfg.allowed_regimes
                and MarketRegime.UNKNOWN not in runtime_cfg.allowed_regimes
            ):
                return False

        return True

    async def evaluate_async(self, context: SignalContext) -> StrategyEvaluation:
        """
        Async wrapper for the event-driven pipeline.

        evaluate() remains synchronous so StrategyEngine can call it in-process.
        If event emission is needed, engine can call evaluate_async().
        """
        evaluation = self.evaluate(context)  # type: ignore[attr-defined]
        await self._emit_evaluation_event(context, evaluation)
        return evaluation

    async def _emit_evaluation_event(
        self,
        context: SignalContext,
        evaluation: StrategyEvaluation,
    ) -> None:
        if self.event_bus is None:
            return

        try:
            signal = getattr(evaluation, "signal", None)
            key = self._resolve_orderflow_key(context)
            scope = orderflow_key_to_dict(key)

            await self.event_bus.emit(
                "strategy.orderflow.evaluated",
                {
                    "strategy_name": self.STRATEGY_NAME,
                    "category": getattr(self.CATEGORY, "value", str(self.CATEGORY)),
                    "exchange": scope["exchange"],
                    "market_type": scope["market_type"],
                    "symbol": scope["symbol"],
                    "timeframe": scope["timeframe"],
                    "key": list(key),
                    "scope": scope,
                    "scope_key": orderflow_key_to_string(key),
                    "timestamp": self._context_timestamp_payload(context),
                    "passed": evaluation.passed,
                    "score": evaluation.score,
                    "confidence": evaluation.confidence,
                    "reasons": list(evaluation.reasons),
                    "signal_id": (
                        getattr(signal, "signal_id", None)
                        if signal is not None
                        else None
                    ),
                    "side": (
                        str(getattr(signal, "side", None))
                        if signal is not None
                        else None
                    ),
                },
                priority=EventPriority.NORMAL,
                source=self.STRATEGY_NAME,
            )

        except Exception:
            self.log_warning(
                "Failed to emit orderflow strategy evaluation event",
                symbol=getattr(context, "symbol", None),
                strategy=self.STRATEGY_NAME,
            )

    # ------------------------------------------------------------------
    # Strategy config helpers
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
    # Scoped orderflow resolution
    # ------------------------------------------------------------------

    def _resolve_orderflow_key(self, context: SignalContext) -> OrderFlowKey:
        """
        Resolve canonical orderflow scope from SignalContext.

        Priority:
        1. context direct attributes;
        2. context.metadata;
        3. context.orderflow["scope"] / metric["scope"];
        4. feature snapshots;
        5. configured defaults.
        """
        orderflow = self._context_orderflow(context)
        metadata = self._context_metadata(context)

        scope_sources: list[Mapping[str, Any]] = []
        for candidate in (
            metadata.get("scope"),
            orderflow.get("scope"),
            orderflow.get("orderflow_scope"),
        ):
            if isinstance(candidate, Mapping):
                scope_sources.append(candidate)

        for metric_name in self.ORDERFLOW_METRIC_NAMES:
            metric_payload = orderflow.get(metric_name)
            if isinstance(metric_payload, Mapping):
                metric_scope = metric_payload.get("scope")
                if isinstance(metric_scope, Mapping):
                    scope_sources.append(metric_scope)
                scope_sources.append(metric_payload)

        direct_source = {
            "exchange": getattr(context, "exchange", None),
            "market_type": getattr(context, "market_type", None),
            "symbol": getattr(context, "symbol", None),
            "timeframe": getattr(context, "timeframe", None),
        }

        feature_source = {
            "exchange": self._feature_value(context, "market.exchange")
            or self._feature_value(context, "orderflow.exchange"),
            "market_type": self._feature_value(context, "market.market_type")
            or self._feature_value(context, "orderflow.market_type"),
            "symbol": self._feature_value(context, "market.symbol")
            or self._feature_value(context, "orderflow.symbol"),
            "timeframe": self._feature_value(context, "market.timeframe")
            or self._feature_value(context, "orderflow.timeframe"),
        }

        symbol = self._first_present(
            direct_source.get("symbol"),
            metadata.get("symbol"),
            orderflow.get("symbol"),
            feature_source.get("symbol"),
            self.DEFAULT_EXCHANGE if False else None,
        )

        if symbol is None:
            symbol = getattr(context, "symbol", None)

        exchange = self._first_present(
            direct_source.get("exchange"),
            metadata.get("exchange"),
            orderflow.get("exchange"),
            feature_source.get("exchange"),
            *(source.get("exchange") for source in scope_sources),
            self.DEFAULT_EXCHANGE,
        )

        market_type = self._first_present(
            direct_source.get("market_type"),
            metadata.get("market_type"),
            orderflow.get("market_type"),
            feature_source.get("market_type"),
            *(source.get("market_type") for source in scope_sources),
            self.DEFAULT_MARKET_TYPE,
        )

        timeframe = self._first_present(
            direct_source.get("timeframe"),
            metadata.get("timeframe"),
            orderflow.get("timeframe"),
            feature_source.get("timeframe"),
            *(source.get("timeframe") for source in scope_sources),
            self.DEFAULT_TIMEFRAME,
        )

        return make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _resolve_orderflow_composite_snapshot(
        self,
        context: SignalContext,
    ) -> OrderflowCompositeSnapshot | None:
        """
        Resolve normalized composite snapshot for continuation/reversal strategies.

        Context payload has priority because StrategyContextBuilder should be the
        canonical source for strategy evaluations. Facade fallback is kept for
        migration and manual strategy usage.
        """
        key = self._resolve_orderflow_key(context)

        context_snapshot = self._build_composite_snapshot_from_context(context, key=key)
        if context_snapshot is not None and context_snapshot.has_minimum_data():
            return context_snapshot

        facade_snapshot = self._build_composite_snapshot_from_facade(context, key=key)
        if facade_snapshot is not None and facade_snapshot.has_minimum_data():
            return facade_snapshot

        return context_snapshot or facade_snapshot

    def _resolve_cvd_stats(self, context: SignalContext) -> Any | None:
        """
        Resolve scoped CVD stats for CvdDivergenceStrategy.

        Concrete CVD strategy can use this method directly during its rewrite.
        """
        key = self._resolve_orderflow_key(context)

        payload = self._extract_metric_payload(context, "cvd")
        if payload:
            return payload

        return self._safe_get_metric_stats_by_key("cvd", key)

    def _build_composite_snapshot_from_context(
        self,
        context: SignalContext,
        *,
        key: OrderFlowKey,
    ) -> OrderflowCompositeSnapshot | None:
        exchange, market_type, symbol, timeframe = key

        cvd = self._extract_metric_payload(context, "cvd")
        volume_delta = self._extract_metric_payload(context, "volume_delta")
        aggressive = self._extract_metric_payload(context, "aggressive_trades")
        imbalance = self._extract_metric_payload(context, "orderbook_imbalance")

        if not any((cvd, volume_delta, aggressive, imbalance)):
            return None

        exchange_symbol = self._coalesce_str(
            self._read(cvd, "exchange_symbol"),
            self._read(volume_delta, "exchange_symbol"),
            self._read(aggressive, "exchange_symbol"),
            self._read(imbalance, "exchange_symbol"),
            symbol,
        )

        timestamp = self._coalesce_float(
            self._read(cvd, "timestamp"),
            self._read(volume_delta, "timestamp"),
            self._read(aggressive, "timestamp"),
            self._read(imbalance, "timestamp"),
            self._context_timestamp_float(context),
        )

        buy_volume = self._coalesce_float(
            self._read(volume_delta, "buy_volume"),
            self._read(cvd, "buy_volume"),
            0.0,
        ) or 0.0
        sell_volume = self._coalesce_float(
            self._read(volume_delta, "sell_volume"),
            self._read(cvd, "sell_volume"),
            0.0,
        ) or 0.0
        buy_notional = self._coalesce_float(
            self._read(volume_delta, "buy_notional"),
            self._read(cvd, "buy_notional"),
            0.0,
        ) or 0.0
        sell_notional = self._coalesce_float(
            self._read(volume_delta, "sell_notional"),
            self._read(cvd, "sell_notional"),
            0.0,
        ) or 0.0

        orderbook_ratio = self._coalesce_float(
            self._read(imbalance, "imbalance_ratio"),
            self._read(imbalance, "ratio"),
            0.0,
        ) or 0.0
        orderbook_diff = self._coalesce_float(
            self._read(imbalance, "imbalance_diff"),
            self._read(imbalance, "diff"),
            self._signed_imbalance_from_ratio(orderbook_ratio),
            0.0,
        ) or 0.0

        return OrderflowCompositeSnapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            timestamp=timestamp,
            source="context",
            last_price=self._coalesce_float(
                self._feature_value(context, "orderflow.last_price"),
                self._feature_value(context, "price.last"),
                self._read(cvd, "last_price"),
                self._read(volume_delta, "last_price"),
                self._read(aggressive, "last_price"),
                self._read(imbalance, "mid_price"),
                getattr(getattr(context, "price", None), "last_price", None),
                getattr(getattr(context, "price", None), "mid_price", None),
            ),
            price_change=self._coalesce_float(
                self._feature_value(context, "orderflow.price_change"),
                self._feature_value(context, "orderflow.cvd.price_change"),
                self._read(cvd, "price_change"),
            ),
            price_change_pct=self._coalesce_float(
                self._feature_value(context, "orderflow.price_change_pct"),
                self._feature_value(context, "orderflow.cvd.price_change_pct"),
                self._feature_value(context, "price.change_pct"),
                self._read(cvd, "price_change_pct"),
                0.0,
            ) or 0.0,
            window_seconds=self._coalesce_float(
                self._read(cvd, "window_seconds"),
                self._read(volume_delta, "window_seconds"),
                self._read(aggressive, "window_seconds"),
                0.0,
            ) or 0.0,
            trades_count=int(
                self._coalesce_int(
                    self._feature_value(context, "orderflow.trades_count"),
                    self._read(cvd, "trades_count"),
                    self._read(volume_delta, "trades_count"),
                    self._read(aggressive, "trades_count"),
                    0,
                )
                or 0
            ),
            total_volume=self._coalesce_float(
                self._feature_value(context, "orderflow.total_volume"),
                self._read(cvd, "total_volume"),
                self._read(volume_delta, "total_volume"),
                self._read(aggressive, "total_volume"),
                buy_volume + sell_volume,
                self._read(aggressive, "aggressive_buy_volume", 0.0)
                + self._read(aggressive, "aggressive_sell_volume", 0.0),
                0.0,
            ) or 0.0,
            total_notional=self._coalesce_float(
                self._feature_value(context, "orderflow.total_notional"),
                self._read(cvd, "total_notional"),
                self._read(volume_delta, "total_notional"),
                buy_notional + sell_notional,
                self._read(aggressive, "aggressive_buy_notional", 0.0)
                + self._read(aggressive, "aggressive_sell_notional", 0.0),
                0.0,
            ) or 0.0,
            cvd_value=self._coalesce_float(
                self._read(cvd, "cvd_value"),
                self._read(cvd, "value"),
                0.0,
            ) or 0.0,
            cvd_open=self._coalesce_float(
                self._read(cvd, "cvd_open"),
                self._read(cvd, "open"),
                0.0,
            ) or 0.0,
            cvd_high=self._coalesce_float(
                self._read(cvd, "cvd_high"),
                self._read(cvd, "high"),
                0.0,
            ) or 0.0,
            cvd_low=self._coalesce_float(
                self._read(cvd, "cvd_low"),
                self._read(cvd, "low"),
                0.0,
            ) or 0.0,
            cvd_close=self._coalesce_float(
                self._read(cvd, "cvd_close"),
                self._read(cvd, "close"),
                0.0,
            ) or 0.0,
            cvd_change=self._coalesce_float(
                self._read(cvd, "cvd_change"),
                self._read(cvd, "change"),
                0.0,
            ) or 0.0,
            cvd_change_pct=self._coalesce_float(
                self._feature_value(context, "orderflow.cvd.change_pct"),
                self._feature_value(context, "orderflow.cvd.cvd_change_pct"),
                self._read(cvd, "cvd_change_pct"),
                self._read(cvd, "change_pct"),
                0.0,
            ) or 0.0,
            cvd_slope=self._coalesce_float(
                self._feature_value(context, "orderflow.cvd.slope"),
                self._feature_value(context, "orderflow.cvd.cvd_slope"),
                self._read(cvd, "cvd_slope"),
                self._read(cvd, "slope"),
                0.0,
            ) or 0.0,
            cvd_delta_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.cvd.delta_ratio"),
                self._read(cvd, "delta_ratio"),
                0.0,
            ) or 0.0,
            cvd_buy_ratio=self._coalesce_float(
                self._read(cvd, "buy_ratio"),
                0.0,
            ) or 0.0,
            cvd_sell_ratio=self._coalesce_float(
                self._read(cvd, "sell_ratio"),
                0.0,
            ) or 0.0,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            volume_delta=self._coalesce_float(
                self._feature_value(context, "orderflow.volume_delta.value"),
                self._feature_value(context, "orderflow.volume_delta.volume_delta"),
                self._read(volume_delta, "volume_delta"),
                self._read(cvd, "volume_delta"),
                0.0,
            ) or 0.0,
            notional_delta=self._coalesce_float(
                self._read(volume_delta, "notional_delta"),
                self._read(cvd, "notional_delta"),
                0.0,
            ) or 0.0,
            volume_delta_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.volume_delta.delta_ratio"),
                self._read(volume_delta, "delta_ratio"),
                0.0,
            ) or 0.0,
            cumulative_volume_delta=self._coalesce_float(
                self._feature_value(context, "orderflow.volume_delta.cumulative_delta"),
                self._feature_value(
                    context,
                    "orderflow.volume_delta.cumulative_volume_delta",
                ),
                self._read(volume_delta, "cumulative_volume_delta"),
                self._read(volume_delta, "cumulative_delta"),
                0.0,
            ) or 0.0,
            cumulative_notional_delta=self._coalesce_float(
                self._read(volume_delta, "cumulative_notional_delta"),
                self._read(volume_delta, "cumulative_notional"),
                0.0,
            ) or 0.0,
            volume_buy_ratio=self._coalesce_float(
                self._read(volume_delta, "buy_ratio"),
                0.0,
            ) or 0.0,
            volume_sell_ratio=self._coalesce_float(
                self._read(volume_delta, "sell_ratio"),
                0.0,
            ) or 0.0,
            avg_trade_size=self._coalesce_float(
                self._read(volume_delta, "avg_trade_size"),
                self._read(cvd, "avg_trade_size"),
                self._read(aggressive, "avg_trade_size"),
                0.0,
            ) or 0.0,
            avg_trade_notional=self._coalesce_float(
                self._read(volume_delta, "avg_trade_notional"),
                self._read(cvd, "avg_trade_notional"),
                self._read(aggressive, "avg_trade_notional"),
                0.0,
            ) or 0.0,
            aggressive_buy_count=int(
                self._coalesce_int(
                    self._read(aggressive, "aggressive_buy_count"),
                    0,
                )
                or 0
            ),
            aggressive_sell_count=int(
                self._coalesce_int(
                    self._read(aggressive, "aggressive_sell_count"),
                    0,
                )
                or 0
            ),
            aggressive_buy_volume=self._coalesce_float(
                self._read(aggressive, "aggressive_buy_volume"),
                0.0,
            ) or 0.0,
            aggressive_sell_volume=self._coalesce_float(
                self._read(aggressive, "aggressive_sell_volume"),
                0.0,
            ) or 0.0,
            aggressive_buy_notional=self._coalesce_float(
                self._read(aggressive, "aggressive_buy_notional"),
                0.0,
            ) or 0.0,
            aggressive_sell_notional=self._coalesce_float(
                self._read(aggressive, "aggressive_sell_notional"),
                0.0,
            ) or 0.0,
            aggressive_net_volume_delta=self._coalesce_float(
                self._read(aggressive, "net_volume_delta"),
                0.0,
            ) or 0.0,
            aggressive_net_notional_delta=self._coalesce_float(
                self._read(aggressive, "net_notional_delta"),
                0.0,
            ) or 0.0,
            aggressive_buy_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.aggressive_trades.buy_ratio"),
                self._read(aggressive, "buy_ratio"),
                self._read(aggressive, "aggressive_buy_ratio"),
                0.0,
            ) or 0.0,
            aggressive_sell_ratio=self._coalesce_float(
                self._feature_value(context, "orderflow.aggressive_trades.sell_ratio"),
                self._read(aggressive, "sell_ratio"),
                self._read(aggressive, "aggressive_sell_ratio"),
                0.0,
            ) or 0.0,
            aggressive_burst_score=self._coalesce_float(
                self._feature_value(context, "orderflow.aggressive_trades.burst_score"),
                self._read(aggressive, "burst_score"),
                0.0,
            ) or 0.0,
            large_buy_trades=int(
                self._coalesce_int(
                    self._feature_value(context, "orderflow.aggressive_trades.large_buy_trades"),
                    self._read(aggressive, "large_buy_trades"),
                    0,
                )
                or 0
            ),
            large_sell_trades=int(
                self._coalesce_int(
                    self._feature_value(context, "orderflow.aggressive_trades.large_sell_trades"),
                    self._read(aggressive, "large_sell_trades"),
                    0,
                )
                or 0
            ),
            aggressive_avg_trade_size=self._coalesce_float(
                self._read(aggressive, "avg_trade_size"),
                0.0,
            ) or 0.0,
            aggressive_avg_trade_notional=self._coalesce_float(
                self._read(aggressive, "avg_trade_notional"),
                0.0,
            ) or 0.0,
            orderbook_bid_volume=self._coalesce_float(
                self._read(imbalance, "bid_volume"),
                0.0,
            ) or 0.0,
            orderbook_ask_volume=self._coalesce_float(
                self._read(imbalance, "ask_volume"),
                0.0,
            ) or 0.0,
            orderbook_imbalance_ratio=orderbook_ratio,
            orderbook_imbalance_diff=orderbook_diff,
            best_bid=self._coalesce_float(self._read(imbalance, "best_bid")),
            best_ask=self._coalesce_float(self._read(imbalance, "best_ask")),
            spread=self._coalesce_float(self._read(imbalance, "spread")),
            mid_price=self._coalesce_float(self._read(imbalance, "mid_price")),
            depth_levels_used=int(
                self._coalesce_int(
                    self._read(imbalance, "depth_levels_used"),
                    0,
                )
                or 0
            ),
            raw_metrics={
                "cvd": cvd,
                "volume_delta": volume_delta,
                "aggressive_trades": aggressive,
                "orderbook_imbalance": imbalance,
            },
            metadata={
                "source": "context",
                "scope": orderflow_key_to_dict(key),
                "scope_key": orderflow_key_to_string(key),
            },
        )

    def _build_composite_snapshot_from_facade(
        self,
        context: SignalContext,
        *,
        key: OrderFlowKey,
    ) -> OrderflowCompositeSnapshot | None:
        facade = self.orderflow_analyzer
        if facade is None:
            return None

        cvd = self._safe_get_metric_stats_by_key("cvd", key)
        volume_delta = self._safe_get_metric_stats_by_key("volume_delta", key)
        aggressive = self._safe_get_metric_stats_by_key("aggressive_trades", key)
        imbalance = self._safe_get_metric_stats_by_key("orderbook_imbalance", key)

        if not any((cvd, volume_delta, aggressive, imbalance)):
            return None

        exchange, market_type, symbol, timeframe = key

        timestamp = self._coalesce_float(
            self._read(cvd, "timestamp"),
            self._read(volume_delta, "timestamp"),
            self._read(aggressive, "timestamp"),
            self._read(imbalance, "timestamp"),
            self._context_timestamp_float(context),
        )

        buy_volume = self._coalesce_float(
            self._read(volume_delta, "buy_volume"),
            self._read(cvd, "buy_volume"),
            0.0,
        ) or 0.0
        sell_volume = self._coalesce_float(
            self._read(volume_delta, "sell_volume"),
            self._read(cvd, "sell_volume"),
            0.0,
        ) or 0.0
        buy_notional = self._coalesce_float(
            self._read(volume_delta, "buy_notional"),
            self._read(cvd, "buy_notional"),
            0.0,
        ) or 0.0
        sell_notional = self._coalesce_float(
            self._read(volume_delta, "sell_notional"),
            self._read(cvd, "sell_notional"),
            0.0,
        ) or 0.0

        orderbook_ratio = self._coalesce_float(
            self._read(imbalance, "imbalance_ratio"),
            0.0,
        ) or 0.0
        orderbook_diff = self._coalesce_float(
            self._read(imbalance, "imbalance_diff"),
            self._signed_imbalance_from_ratio(orderbook_ratio),
            0.0,
        ) or 0.0

        return OrderflowCompositeSnapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=self._coalesce_str(
                self._read(cvd, "exchange_symbol"),
                self._read(volume_delta, "exchange_symbol"),
                self._read(aggressive, "exchange_symbol"),
                self._read(imbalance, "exchange_symbol"),
                symbol,
            ),
            timestamp=timestamp,
            source="facade",
            last_price=self._coalesce_float(
                self._read(cvd, "last_price"),
                self._read(volume_delta, "last_price"),
                self._read(aggressive, "last_price"),
                self._read(imbalance, "mid_price"),
                getattr(getattr(context, "price", None), "last_price", None),
                getattr(getattr(context, "price", None), "mid_price", None),
            ),
            price_change=self._coalesce_float(self._read(cvd, "price_change")),
            price_change_pct=self._coalesce_float(
                self._read(cvd, "price_change_pct"),
                0.0,
            ) or 0.0,
            window_seconds=self._coalesce_float(
                self._read(cvd, "window_seconds"),
                self._read(volume_delta, "window_seconds"),
                self._read(aggressive, "window_seconds"),
                0.0,
            ) or 0.0,
            trades_count=int(
                self._coalesce_int(
                    self._read(cvd, "trades_count"),
                    self._read(volume_delta, "trades_count"),
                    self._read(aggressive, "trades_count"),
                    0,
                )
                or 0
            ),
            total_volume=self._coalesce_float(
                buy_volume + sell_volume,
                self._read(aggressive, "aggressive_buy_volume", 0.0)
                + self._read(aggressive, "aggressive_sell_volume", 0.0),
                0.0,
            ) or 0.0,
            total_notional=self._coalesce_float(
                buy_notional + sell_notional,
                self._read(aggressive, "aggressive_buy_notional", 0.0)
                + self._read(aggressive, "aggressive_sell_notional", 0.0),
                0.0,
            ) or 0.0,
            cvd_value=self._coalesce_float(self._read(cvd, "cvd_value"), 0.0) or 0.0,
            cvd_open=self._coalesce_float(self._read(cvd, "cvd_open"), 0.0) or 0.0,
            cvd_high=self._coalesce_float(self._read(cvd, "cvd_high"), 0.0) or 0.0,
            cvd_low=self._coalesce_float(self._read(cvd, "cvd_low"), 0.0) or 0.0,
            cvd_close=self._coalesce_float(self._read(cvd, "cvd_close"), 0.0) or 0.0,
            cvd_change=self._coalesce_float(self._read(cvd, "cvd_change"), 0.0) or 0.0,
            cvd_change_pct=self._coalesce_float(
                self._read(cvd, "cvd_change_pct"),
                0.0,
            ) or 0.0,
            cvd_slope=self._coalesce_float(self._read(cvd, "cvd_slope"), 0.0) or 0.0,
            cvd_delta_ratio=self._coalesce_float(self._read(cvd, "delta_ratio"), 0.0) or 0.0,
            cvd_buy_ratio=self._coalesce_float(self._read(cvd, "buy_ratio"), 0.0) or 0.0,
            cvd_sell_ratio=self._coalesce_float(self._read(cvd, "sell_ratio"), 0.0) or 0.0,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            volume_delta=self._coalesce_float(
                self._read(volume_delta, "volume_delta"),
                self._read(cvd, "volume_delta"),
                0.0,
            ) or 0.0,
            notional_delta=self._coalesce_float(
                self._read(volume_delta, "notional_delta"),
                self._read(cvd, "notional_delta"),
                0.0,
            ) or 0.0,
            volume_delta_ratio=self._coalesce_float(
                self._read(volume_delta, "delta_ratio"),
                0.0,
            ) or 0.0,
            cumulative_volume_delta=self._coalesce_float(
                self._read(volume_delta, "cumulative_volume_delta"),
                0.0,
            ) or 0.0,
            cumulative_notional_delta=self._coalesce_float(
                self._read(volume_delta, "cumulative_notional_delta"),
                0.0,
            ) or 0.0,
            volume_buy_ratio=self._coalesce_float(
                self._read(volume_delta, "buy_ratio"),
                0.0,
            ) or 0.0,
            volume_sell_ratio=self._coalesce_float(
                self._read(volume_delta, "sell_ratio"),
                0.0,
            ) or 0.0,
            avg_trade_size=self._coalesce_float(
                self._read(volume_delta, "avg_trade_size"),
                self._read(cvd, "avg_trade_size"),
                self._read(aggressive, "avg_trade_size"),
                0.0,
            ) or 0.0,
            avg_trade_notional=self._coalesce_float(
                self._read(volume_delta, "avg_trade_notional"),
                self._read(cvd, "avg_trade_notional"),
                self._read(aggressive, "avg_trade_notional"),
                0.0,
            ) or 0.0,
            aggressive_buy_count=int(
                self._coalesce_int(self._read(aggressive, "aggressive_buy_count"), 0)
                or 0
            ),
            aggressive_sell_count=int(
                self._coalesce_int(self._read(aggressive, "aggressive_sell_count"), 0)
                or 0
            ),
            aggressive_buy_volume=self._coalesce_float(
                self._read(aggressive, "aggressive_buy_volume"),
                0.0,
            ) or 0.0,
            aggressive_sell_volume=self._coalesce_float(
                self._read(aggressive, "aggressive_sell_volume"),
                0.0,
            ) or 0.0,
            aggressive_buy_notional=self._coalesce_float(
                self._read(aggressive, "aggressive_buy_notional"),
                0.0,
            ) or 0.0,
            aggressive_sell_notional=self._coalesce_float(
                self._read(aggressive, "aggressive_sell_notional"),
                0.0,
            ) or 0.0,
            aggressive_net_volume_delta=self._coalesce_float(
                self._read(aggressive, "net_volume_delta"),
                0.0,
            ) or 0.0,
            aggressive_net_notional_delta=self._coalesce_float(
                self._read(aggressive, "net_notional_delta"),
                0.0,
            ) or 0.0,
            aggressive_buy_ratio=self._coalesce_float(
                self._read(aggressive, "buy_ratio"),
                0.0,
            ) or 0.0,
            aggressive_sell_ratio=self._coalesce_float(
                self._read(aggressive, "sell_ratio"),
                0.0,
            ) or 0.0,
            aggressive_burst_score=self._coalesce_float(
                self._read(aggressive, "burst_score"),
                0.0,
            ) or 0.0,
            large_buy_trades=int(
                self._coalesce_int(self._read(aggressive, "large_buy_trades"), 0)
                or 0
            ),
            large_sell_trades=int(
                self._coalesce_int(self._read(aggressive, "large_sell_trades"), 0)
                or 0
            ),
            aggressive_avg_trade_size=self._coalesce_float(
                self._read(aggressive, "avg_trade_size"),
                0.0,
            ) or 0.0,
            aggressive_avg_trade_notional=self._coalesce_float(
                self._read(aggressive, "avg_trade_notional"),
                0.0,
            ) or 0.0,
            orderbook_bid_volume=self._coalesce_float(
                self._read(imbalance, "bid_volume"),
                0.0,
            ) or 0.0,
            orderbook_ask_volume=self._coalesce_float(
                self._read(imbalance, "ask_volume"),
                0.0,
            ) or 0.0,
            orderbook_imbalance_ratio=orderbook_ratio,
            orderbook_imbalance_diff=orderbook_diff,
            best_bid=self._coalesce_float(self._read(imbalance, "best_bid")),
            best_ask=self._coalesce_float(self._read(imbalance, "best_ask")),
            spread=self._coalesce_float(self._read(imbalance, "spread")),
            mid_price=self._coalesce_float(self._read(imbalance, "mid_price")),
            depth_levels_used=int(
                self._coalesce_int(self._read(imbalance, "depth_levels_used"), 0)
                or 0
            ),
            raw_metrics={
                "cvd": cvd,
                "volume_delta": volume_delta,
                "aggressive_trades": aggressive,
                "orderbook_imbalance": imbalance,
            },
            metadata={
                "source": "facade",
                "scope": orderflow_key_to_dict(key),
                "scope_key": orderflow_key_to_string(key),
            },
        )

    # ------------------------------------------------------------------
    # Analytics facade/module access
    # ------------------------------------------------------------------

    def _safe_get_metric_stats_by_key(
        self,
        module_name: str,
        key: OrderFlowKey,
    ) -> Any | None:
        facade = self.orderflow_analyzer
        if facade is None:
            return None

        # 1. Prefer facade aggregate scoped access.
        try:
            getter = getattr(facade, "get_latest_stats_by_key", None)
            if callable(getter):
                result = getter(key)
                metric_result = self._extract_metric_from_result(result, module_name)
                if metric_result is not None:
                    return metric_result
        except Exception:
            self.log_warning(
                "Failed to resolve orderflow stats through facade.get_latest_stats_by_key",
                strategy=self.STRATEGY_NAME,
                module=module_name,
                scope_key=orderflow_key_to_string(key),
            )

        # 2. Prefer module scoped access.
        module = self._get_orderflow_module(facade, module_name)
        if module is not None:
            try:
                getter = getattr(module, "get_latest_stats_by_key", None)
                if callable(getter):
                    result = getter(key)
                    if result is not None:
                        return result
            except Exception:
                self.log_warning(
                    "Failed to resolve orderflow stats through module.get_latest_stats_by_key",
                    strategy=self.STRATEGY_NAME,
                    module=module_name,
                    scope_key=orderflow_key_to_string(key),
                )

        # 3. Try facade keyword scoped API.
        try:
            exchange, market_type, symbol, timeframe = key
            getter = getattr(facade, "get_latest_stats", None)
            if callable(getter):
                result = getter(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                metric_result = self._extract_metric_from_result(result, module_name)
                if metric_result is not None:
                    return metric_result
        except TypeError:
            # Old facade signatures are handled below.
            pass
        except Exception:
            self.log_warning(
                "Failed to resolve orderflow stats through facade.get_latest_stats keyword API",
                strategy=self.STRATEGY_NAME,
                module=module_name,
                scope_key=orderflow_key_to_string(key),
            )

        # 4. Temporary migration fallback.
        if self.ALLOW_LEGACY_SYMBOL_FALLBACK:
            return self._safe_get_latest_stats(
                facade=facade,
                module_name=module_name,
                symbol=key[2],
            )

        return None

    @staticmethod
    def _safe_get_latest_stats(
        facade: OrderFlowAnalyzer,
        module_name: str,
        symbol: str,
    ) -> Any:
        """
        Backward-compatible symbol-only resolver.

        New strategy code should not call this directly. Use
        _safe_get_metric_stats_by_key(module_name, key) instead.
        """
        module = getattr(facade, module_name, None)

        if module is None and hasattr(facade, "get_module"):
            module = facade.get_module(module_name)

        if module is not None:
            getter = getattr(module, "get_latest_stats", None)
            if callable(getter):
                return getter(symbol)

        getter = getattr(facade, "get_latest_stats", None)
        if callable(getter):
            result = getter(symbol)
            if isinstance(result, Mapping):
                return result.get(module_name)
            return result

        return None

    @staticmethod
    def _get_orderflow_module(facade: OrderFlowAnalyzer, module_name: str) -> Any | None:
        module = getattr(facade, module_name, None)

        if module is None and hasattr(facade, "get_module"):
            module = facade.get_module(module_name)

        return module

    @staticmethod
    def _extract_metric_from_result(result: Any, module_name: str) -> Any | None:
        if result is None:
            return None

        if isinstance(result, Mapping):
            direct = result.get(module_name)
            if direct is not None:
                return direct

            # Some facades may wrap values into {"metrics": {...}}.
            metrics = result.get("metrics")
            if isinstance(metrics, Mapping):
                metric_result = metrics.get(module_name)
                if metric_result is not None:
                    return metric_result

            return None

        metric = getattr(result, "metric", None)
        metric_value = getattr(metric, "value", metric)
        if str(metric_value) == module_name:
            return result

        return result

    # ------------------------------------------------------------------
    # Context payload extraction
    # ------------------------------------------------------------------

    def _extract_metric_payload(
        self,
        context: SignalContext,
        metric_name: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        orderflow = self._context_orderflow(context)
        raw_metric = orderflow.get(metric_name)

        if isinstance(raw_metric, Mapping):
            payload.update(dict(raw_metric))
        elif raw_metric is not None:
            payload.update(self._model_to_plain_dict(raw_metric))

        feature_aliases = self._metric_feature_aliases(metric_name)
        for target_name, aliases in feature_aliases.items():
            if target_name in payload:
                continue

            for alias in aliases:
                value = self._feature_value(context, alias)
                if value is not None:
                    payload[target_name] = value
                    break

        return payload

    @staticmethod
    def _metric_feature_aliases(metric_name: str) -> dict[str, list[str]]:
        common_scope = {
            "exchange": [f"orderflow.{metric_name}.exchange", "orderflow.exchange"],
            "market_type": [
                f"orderflow.{metric_name}.market_type",
                "orderflow.market_type",
            ],
            "symbol": [f"orderflow.{metric_name}.symbol", "orderflow.symbol"],
            "timeframe": [
                f"orderflow.{metric_name}.timeframe",
                "orderflow.timeframe",
            ],
            "exchange_symbol": [f"orderflow.{metric_name}.exchange_symbol"],
            "timestamp": [f"orderflow.{metric_name}.timestamp"],
            "window_seconds": [f"orderflow.{metric_name}.window_seconds"],
            "trades_count": [
                f"orderflow.{metric_name}.trades_count",
                "orderflow.trades_count",
            ],
        }

        aliases: dict[str, dict[str, list[str]]] = {
            "cvd": {
                **common_scope,
                "buy_volume": ["orderflow.cvd.buy_volume"],
                "sell_volume": ["orderflow.cvd.sell_volume"],
                "volume_delta": ["orderflow.cvd.volume_delta"],
                "buy_notional": ["orderflow.cvd.buy_notional"],
                "sell_notional": ["orderflow.cvd.sell_notional"],
                "notional_delta": ["orderflow.cvd.notional_delta"],
                "cvd_value": ["orderflow.cvd.value", "orderflow.cvd.cvd_value"],
                "cvd_open": ["orderflow.cvd.open", "orderflow.cvd.cvd_open"],
                "cvd_high": ["orderflow.cvd.high", "orderflow.cvd.cvd_high"],
                "cvd_low": ["orderflow.cvd.low", "orderflow.cvd.cvd_low"],
                "cvd_close": ["orderflow.cvd.close", "orderflow.cvd.cvd_close"],
                "cvd_change": ["orderflow.cvd.change", "orderflow.cvd.cvd_change"],
                "cvd_change_pct": [
                    "orderflow.cvd.change_pct",
                    "orderflow.cvd.cvd_change_pct",
                ],
                "cvd_slope": ["orderflow.cvd.slope", "orderflow.cvd.cvd_slope"],
                "delta_ratio": ["orderflow.cvd.delta_ratio"],
                "buy_ratio": ["orderflow.cvd.buy_ratio"],
                "sell_ratio": ["orderflow.cvd.sell_ratio"],
                "avg_trade_size": ["orderflow.cvd.avg_trade_size"],
                "avg_trade_notional": ["orderflow.cvd.avg_trade_notional"],
                "last_price": ["orderflow.cvd.last_price"],
                "price_change": ["orderflow.cvd.price_change"],
                "price_change_pct": ["orderflow.cvd.price_change_pct"],
            },
            "volume_delta": {
                **common_scope,
                "buy_volume": ["orderflow.volume_delta.buy_volume"],
                "sell_volume": ["orderflow.volume_delta.sell_volume"],
                "buy_notional": ["orderflow.volume_delta.buy_notional"],
                "sell_notional": ["orderflow.volume_delta.sell_notional"],
                "volume_delta": [
                    "orderflow.volume_delta.value",
                    "orderflow.volume_delta.volume_delta",
                ],
                "notional_delta": ["orderflow.volume_delta.notional_delta"],
                "delta_ratio": ["orderflow.volume_delta.delta_ratio"],
                "cumulative_volume_delta": [
                    "orderflow.volume_delta.cumulative_delta",
                    "orderflow.volume_delta.cumulative_volume_delta",
                ],
                "cumulative_notional_delta": [
                    "orderflow.volume_delta.cumulative_notional_delta",
                ],
                "buy_ratio": ["orderflow.volume_delta.buy_ratio"],
                "sell_ratio": ["orderflow.volume_delta.sell_ratio"],
                "avg_trade_size": ["orderflow.volume_delta.avg_trade_size"],
                "avg_trade_notional": ["orderflow.volume_delta.avg_trade_notional"],
                "last_price": ["orderflow.volume_delta.last_price"],
            },
            "aggressive_trades": {
                **common_scope,
                "aggressive_buy_count": [
                    "orderflow.aggressive_trades.aggressive_buy_count",
                ],
                "aggressive_sell_count": [
                    "orderflow.aggressive_trades.aggressive_sell_count",
                ],
                "aggressive_buy_volume": [
                    "orderflow.aggressive_trades.aggressive_buy_volume",
                ],
                "aggressive_sell_volume": [
                    "orderflow.aggressive_trades.aggressive_sell_volume",
                ],
                "aggressive_buy_notional": [
                    "orderflow.aggressive_trades.aggressive_buy_notional",
                ],
                "aggressive_sell_notional": [
                    "orderflow.aggressive_trades.aggressive_sell_notional",
                ],
                "net_volume_delta": [
                    "orderflow.aggressive_trades.net_volume_delta",
                ],
                "net_notional_delta": [
                    "orderflow.aggressive_trades.net_notional_delta",
                ],
                "buy_ratio": ["orderflow.aggressive_trades.buy_ratio"],
                "sell_ratio": ["orderflow.aggressive_trades.sell_ratio"],
                "burst_score": ["orderflow.aggressive_trades.burst_score"],
                "large_buy_trades": [
                    "orderflow.aggressive_trades.large_buy_trades",
                ],
                "large_sell_trades": [
                    "orderflow.aggressive_trades.large_sell_trades",
                ],
                "avg_trade_size": ["orderflow.aggressive_trades.avg_trade_size"],
                "avg_trade_notional": [
                    "orderflow.aggressive_trades.avg_trade_notional",
                ],
                "last_price": ["orderflow.aggressive_trades.last_price"],
            },
            "orderbook_imbalance": {
                **common_scope,
                "bid_volume": ["orderflow.orderbook_imbalance.bid_volume"],
                "ask_volume": ["orderflow.orderbook_imbalance.ask_volume"],
                "imbalance_ratio": [
                    "orderflow.orderbook_imbalance.ratio",
                    "orderflow.orderbook_imbalance.imbalance_ratio",
                ],
                "imbalance_diff": [
                    "orderflow.orderbook_imbalance.diff",
                    "orderflow.orderbook_imbalance.imbalance_diff",
                ],
                "best_bid": ["orderflow.orderbook_imbalance.best_bid"],
                "best_ask": ["orderflow.orderbook_imbalance.best_ask"],
                "spread": ["orderflow.orderbook_imbalance.spread"],
                "mid_price": ["orderflow.orderbook_imbalance.mid_price"],
                "depth_levels_used": [
                    "orderflow.orderbook_imbalance.depth_levels_used",
                ],
            },
        }

        return aliases.get(metric_name, common_scope)

    def _context_orderflow(self, context: SignalContext) -> dict[str, Any]:
        orderflow = getattr(context, "orderflow", None)
        if isinstance(orderflow, Mapping):
            return dict(orderflow)
        return {}

    def _context_metadata(self, context: SignalContext) -> dict[str, Any]:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, Mapping):
            return dict(metadata)
        return {}

    # ------------------------------------------------------------------
    # Generic value helpers
    # ------------------------------------------------------------------

    def _resolve_reference_price(self, context: SignalContext, data: Any) -> float | None:
        if context.price is not None:
            if context.price.mid_price is not None:
                return context.price.mid_price
            if context.price.last_price is not None:
                return context.price.last_price

        return self._coalesce_float(getattr(data, "last_price", None))

    def _feature_value(self, context: SignalContext, name: str) -> Any:
        getter = getattr(context, "get_feature_snapshot", None)
        if not callable(getter):
            return None

        snapshot = getter(name)
        if snapshot is None:
            return None

        return getattr(snapshot, "value", None)

    @staticmethod
    def _read(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default

        if isinstance(obj, Mapping):
            return obj.get(name, default)

        return getattr(obj, name, default)

    @staticmethod
    def _model_to_plain_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}

        if isinstance(obj, Mapping):
            return dict(obj)

        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                result = to_dict()
                return dict(result) if isinstance(result, Mapping) else {}
            except Exception:
                return {}

        result: dict[str, Any] = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if callable(value):
                continue
            result[name] = value

        return result

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    @staticmethod
    def _coalesce_str(*values: Any) -> str | None:
        for value in values:
            if value is None:
                continue
            parsed = str(value).strip()
            if parsed:
                return parsed
        return None

    @staticmethod
    def _coalesce_float(*values: Any) -> float | None:
        for value in values:
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
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
    def _normalize_percent(value: float, *, scale: float = 2.0) -> float:
        if scale <= 0:
            return 0.0

        return max(0.0, min(abs(value) / scale, 1.0))

    @staticmethod
    def _normalize_ratio(value: float, *, scale: float = 1.0) -> float:
        if scale <= 0:
            return 0.0

        return max(0.0, min(abs(value) / scale, 1.0))

    @staticmethod
    def _normalize_magnitude(value: float, *, scale: float = 10.0) -> float:
        if value <= 0 or scale <= 0:
            return 0.0

        return max(0.0, min(value / scale, 1.0))

    @staticmethod
    def _signed_imbalance_from_ratio(value: Any) -> float:
        parsed = _safe_float(value)
        if parsed is None:
            return 0.0

        if -1.0 <= parsed <= 1.0 and parsed < 0.0:
            return parsed

        # If analytics ratio is 0..1, convert bid ratio to signed diff.
        if 0.0 <= parsed <= 1.0:
            return (parsed * 2.0) - 1.0

        return parsed

    @staticmethod
    def _enum_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

    @staticmethod
    def _context_timestamp_float(context: SignalContext) -> float | None:
        timestamp = getattr(context, "timestamp", None)

        if isinstance(timestamp, datetime):
            return timestamp.timestamp()

        return _safe_float(timestamp)

    @staticmethod
    def _context_timestamp_payload(context: SignalContext) -> Any:
        timestamp = getattr(context, "timestamp", None)

        if hasattr(timestamp, "isoformat"):
            return timestamp.isoformat()

        return timestamp

    @staticmethod
    def _is_fresh_timestamp(
        timestamp: float | None,
        *,
        max_age_seconds: float,
    ) -> bool:
        if timestamp is None:
            return False

        if max_age_seconds <= 0:
            return True

        return (time.time() - timestamp) <= max_age_seconds


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result