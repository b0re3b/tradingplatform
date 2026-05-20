"""
Backtest model analytics.

This module analyzes the quality of the whole trading system after a backtest:
signals, strategy attribution, regimes, features, risk decisions and execution
quality.

Important:
- No EventBus usage here.
- No strategy/risk/execution decisions here.
- No live exchange calls here.
- This module only analyzes completed backtest artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable

from backtesting.config import ModelAnalyticsConfig
from backtesting.enums import SignalOutcome
from backtesting.models import (
    BacktestExecutionRecord,
    BacktestModelAnalytics,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    RegimePerformanceStats,
    SignalQualityStats,
    SimulatedFill,
    SimulatedOrder,
    SimulatedPosition,
    SimulatedTrade,
    StrategyAttribution,
    safe_div,
)


@dataclass(slots=True)
class ModelAnalyticsInput:
    """
    Input bundle for model analytics.
    """

    signals: list[BacktestSignalRecord] = field(default_factory=list)
    risk_decisions: list[BacktestRiskDecisionRecord] = field(default_factory=list)
    orders: list[SimulatedOrder] = field(default_factory=list)
    fills: list[SimulatedFill] = field(default_factory=list)
    positions: list[SimulatedPosition] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)
    execution_records: list[BacktestExecutionRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignalAttributionRecord:
    """
    Signal-to-trade attribution record.
    """

    signal_id: str
    strategy_name: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    outcome: SignalOutcome = SignalOutcome.UNKNOWN
    trade_id: str | None = None
    position_id: str | None = None
    net_pnl: float = 0.0
    r_multiple: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RiskDecisionAnalytics:
    """
    Diagnostics for risk decisions.
    """

    total_decisions: int = 0
    approved: int = 0
    blocked: int = 0
    approval_rate: float = 0.0
    block_rate: float = 0.0
    blocked_reason_counts: dict[str, int] = field(default_factory=dict)

    approved_signals_pnl: float = 0.0
    blocked_signals_count: int = 0

    avg_final_size: float = 0.0
    avg_final_leverage: float = 0.0
    avg_final_margin: float = 0.0
    avg_final_notional: float = 0.0

    max_final_size: float = 0.0
    max_final_leverage: float = 0.0
    max_final_margin: float = 0.0
    max_final_notional: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "approved": self.approved,
            "blocked": self.blocked,
            "approval_rate": self.approval_rate,
            "block_rate": self.block_rate,
            "blocked_reason_counts": dict(self.blocked_reason_counts),
            "approved_signals_pnl": self.approved_signals_pnl,
            "blocked_signals_count": self.blocked_signals_count,
            "avg_final_size": self.avg_final_size,
            "avg_final_leverage": self.avg_final_leverage,
            "avg_final_margin": self.avg_final_margin,
            "avg_final_notional": self.avg_final_notional,
            "max_final_size": self.max_final_size,
            "max_final_leverage": self.max_final_leverage,
            "max_final_margin": self.max_final_margin,
            "max_final_notional": self.max_final_notional,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ExecutionQualityAnalytics:
    """
    Diagnostics for simulated execution quality.
    """

    orders: int = 0
    fills: int = 0
    rejected_orders: int = 0
    cancelled_orders: int = 0

    fill_rate: float = 0.0
    rejection_rate: float = 0.0
    cancellation_rate: float = 0.0

    avg_slippage: float = 0.0
    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    avg_latency_ms: float = 0.0

    rejected_reason_counts: dict[str, int] = field(default_factory=dict)
    by_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "orders": self.orders,
            "fills": self.fills,
            "rejected_orders": self.rejected_orders,
            "cancelled_orders": self.cancelled_orders,
            "fill_rate": self.fill_rate,
            "rejection_rate": self.rejection_rate,
            "cancellation_rate": self.cancellation_rate,
            "avg_slippage": self.avg_slippage,
            "avg_slippage_bps": self.avg_slippage_bps,
            "max_slippage_bps": self.max_slippage_bps,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "avg_latency_ms": self.avg_latency_ms,
            "rejected_reason_counts": dict(self.rejected_reason_counts),
            "by_strategy": self.by_strategy,
            "by_symbol": self.by_symbol,
            "metadata": self.metadata,
        }


class SignalQualityAnalyzer:
    """
    Analyzes signal lifecycle quality and signal-to-trade conversion.
    """

    def __init__(self, config: ModelAnalyticsConfig | None = None) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

    def analyze(
        self,
        *,
        signals: list[BacktestSignalRecord],
        trades: list[SimulatedTrade],
        risk_decisions: list[BacktestRiskDecisionRecord],
    ) -> SignalQualityStats:
        stats = SignalQualityStats()

        if not signals:
            return stats

        attribution = self.attribute_signals_to_trades(signals=signals, trades=trades)

        stats.signals_generated = len(signals)
        stats.signals_confirmed = len([
            signal for signal in signals
            if signal.outcome in {
                SignalOutcome.CONFIRMED_BY_RISK,
                SignalOutcome.ORDER_FILLED,
                SignalOutcome.POSITION_OPENED,
                SignalOutcome.POSITION_CLOSED_WIN,
                SignalOutcome.POSITION_CLOSED_LOSS,
                SignalOutcome.POSITION_CLOSED_BREAKEVEN,
            }
        ])

        if risk_decisions:
            stats.signals_confirmed = len([item for item in risk_decisions if item.approved])
            stats.signals_blocked_by_risk = len([
                item for item in risk_decisions
                if item.blocked or not item.approved
            ])
        else:
            stats.signals_blocked_by_risk = len([
                signal for signal in signals
                if signal.outcome == SignalOutcome.BLOCKED_BY_RISK
            ])

        stats.signals_executed = len([
            item for item in attribution
            if item.trade_id is not None or item.position_id is not None
        ])

        stats.signals_profitable = len([
            item for item in attribution
            if item.net_pnl > 0
        ])
        stats.signals_unprofitable = len([
            item for item in attribution
            if item.net_pnl < 0
        ])

        stats.confirmation_rate = safe_div(stats.signals_confirmed, stats.signals_generated) * 100.0
        stats.execution_rate = safe_div(stats.signals_executed, stats.signals_generated) * 100.0
        stats.profitable_signal_rate = safe_div(stats.signals_profitable, stats.signals_executed) * 100.0

        executed_pnls = [item.net_pnl for item in attribution if item.trade_id is not None]
        stats.average_signal_pnl = mean(executed_pnls) if executed_pnls else 0.0

        r_values = [
            item.r_multiple for item in attribution
            if item.r_multiple is not None
        ]
        stats.average_signal_r = mean(r_values) if r_values else None

        return stats

    def attribute_signals_to_trades(
        self,
        *,
        signals: list[BacktestSignalRecord],
        trades: list[SimulatedTrade],
    ) -> list[SignalAttributionRecord]:
        trade_by_signal: dict[str, list[SimulatedTrade]] = defaultdict(list)

        for trade in trades:
            if trade.signal_id:
                trade_by_signal[trade.signal_id].append(trade)

        records: list[SignalAttributionRecord] = []

        for signal in signals:
            if not signal.signal_id:
                records.append(
                    SignalAttributionRecord(
                        signal_id="",
                        strategy_name=signal.strategy_name,
                        symbol=signal.symbol,
                        timeframe=signal.timeframe,
                        outcome=signal.outcome,
                        metadata={"warning": "signal_id_missing"},
                    )
                )
                continue

            linked_trades = trade_by_signal.get(signal.signal_id, [])

            if not linked_trades:
                records.append(
                    SignalAttributionRecord(
                        signal_id=signal.signal_id,
                        strategy_name=signal.strategy_name,
                        symbol=signal.symbol,
                        timeframe=signal.timeframe,
                        outcome=signal.outcome,
                        net_pnl=signal.pnl,
                        r_multiple=signal.r_multiple,
                    )
                )
                continue

            net_pnl = sum(trade.net_pnl for trade in linked_trades)
            r_values = [
                trade.r_multiple for trade in linked_trades
                if trade.r_multiple is not None
            ]

            first_trade = linked_trades[0]

            records.append(
                SignalAttributionRecord(
                    signal_id=signal.signal_id,
                    strategy_name=signal.strategy_name or first_trade.strategy_name,
                    symbol=signal.symbol or first_trade.symbol,
                    timeframe=signal.timeframe,
                    outcome=self._outcome_from_trades(linked_trades),
                    trade_id=first_trade.trade_id,
                    position_id=first_trade.position_id,
                    net_pnl=net_pnl,
                    r_multiple=mean(r_values) if r_values else None,
                    metadata={
                        "linked_trades": len(linked_trades),
                    },
                )
            )

        return records

    @staticmethod
    def _outcome_from_trades(trades: list[SimulatedTrade]) -> SignalOutcome:
        net_pnl = sum(trade.net_pnl for trade in trades)

        if net_pnl > 0:
            return SignalOutcome.POSITION_CLOSED_WIN

        if net_pnl < 0:
            return SignalOutcome.POSITION_CLOSED_LOSS

        return SignalOutcome.POSITION_CLOSED_BREAKEVEN


class StrategyAttributionAnalyzer:
    """
    Analyzes which strategies contribute to system results.
    """

    def __init__(self, config: ModelAnalyticsConfig | None = None) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

    def analyze(
        self,
        *,
        trades: list[SimulatedTrade],
        signals: list[BacktestSignalRecord],
        risk_decisions: list[BacktestRiskDecisionRecord],
    ) -> list[StrategyAttribution]:
        strategy_names = sorted(
            {
                trade.strategy_name
                for trade in trades
                if trade.strategy_name
            }
            | {
                signal.strategy_name
                for signal in signals
                if signal.strategy_name
            }
            | {
                decision.strategy_name
                for decision in risk_decisions
                if decision.strategy_name
            }
        )

        if not strategy_names:
            return []

        total_net_profit = sum(
            trade.net_pnl
            for trade in trades
            if trade.is_closed
        )

        result: list[StrategyAttribution] = []

        for strategy_name in strategy_names:
            strategy_trades = [
                trade for trade in trades
                if trade.strategy_name == strategy_name and trade.is_closed
            ]
            strategy_signals = [
                signal for signal in signals
                if signal.strategy_name == strategy_name
            ]
            strategy_risk_decisions = [
                decision for decision in risk_decisions
                if decision.strategy_name == strategy_name
            ]

            net_profit = sum(trade.net_pnl for trade in strategy_trades)
            wins = len([trade for trade in strategy_trades if trade.net_pnl > 0])
            losses = len([trade for trade in strategy_trades if trade.net_pnl < 0])

            gross_profit = sum(
                trade.net_pnl for trade in strategy_trades
                if trade.net_pnl > 0
            )
            gross_loss = abs(
                sum(
                    trade.net_pnl for trade in strategy_trades
                    if trade.net_pnl < 0
                )
            )

            blocked_signals = len([
                decision for decision in strategy_risk_decisions
                if decision.blocked or not decision.approved
            ])

            attribution = StrategyAttribution(
                strategy_name=strategy_name,
                net_profit=net_profit,
                profit_share_pct=safe_div(net_profit, total_net_profit) * 100.0 if total_net_profit else 0.0,
                drawdown_contribution=self._estimate_drawdown_contribution(strategy_trades),
                trades=len(strategy_trades),
                win_rate=safe_div(wins, wins + losses) * 100.0,
                profit_factor=safe_div(gross_profit, gross_loss),
                expectancy=self._calculate_expectancy(strategy_trades),
                signals=len(strategy_signals),
                blocked_signals=blocked_signals,
                metadata={
                    "gross_profit": gross_profit,
                    "gross_loss": gross_loss,
                    "wins": wins,
                    "losses": losses,
                    "breakeven": len([
                        trade for trade in strategy_trades
                        if trade.net_pnl == 0
                    ]),
                    "min_trades_required": self.config.min_trades_for_strategy_stats,
                    "enough_trades": len(strategy_trades) >= self.config.min_trades_for_strategy_stats,
                },
            )
            result.append(attribution)

        result.sort(key=lambda item: item.net_profit, reverse=True)
        return result

    @staticmethod
    def _calculate_expectancy(trades: list[SimulatedTrade]) -> float:
        if not trades:
            return 0.0

        wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
        losses = [trade.net_pnl for trade in trades if trade.net_pnl < 0]

        win_rate = safe_div(len(wins), len(trades))
        loss_rate = safe_div(len(losses), len(trades))

        avg_win = mean(wins) if wins else 0.0
        avg_loss = abs(mean(losses)) if losses else 0.0

        return win_rate * avg_win - loss_rate * avg_loss

    @staticmethod
    def _estimate_drawdown_contribution(trades: list[SimulatedTrade]) -> float:
        """
        Simple attribution proxy: sum of losing trades.
        """

        return abs(sum(trade.net_pnl for trade in trades if trade.net_pnl < 0))


class RegimePerformanceAnalyzer:
    """
    Analyzes performance grouped by market regime.

    Regime is extracted from trade.metadata, signal.metadata or payload metadata.
    """

    def __init__(self, config: ModelAnalyticsConfig | None = None) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

    def analyze(
        self,
        *,
        trades: list[SimulatedTrade],
        signals: list[BacktestSignalRecord],
    ) -> list[RegimePerformanceStats]:
        grouped: dict[str, list[SimulatedTrade]] = defaultdict(list)

        signal_regime_by_id = self._signal_regime_map(signals)

        for trade in trades:
            regime = self._extract_trade_regime(trade)

            if regime is None and trade.signal_id:
                regime = signal_regime_by_id.get(trade.signal_id)

            if regime is None:
                regime = "unknown"

            grouped[regime].append(trade)

        result: list[RegimePerformanceStats] = []

        for regime, regime_trades in grouped.items():
            closed = [trade for trade in regime_trades if trade.is_closed]
            if not closed:
                continue

            wins = [trade for trade in closed if trade.net_pnl > 0]
            losses = [trade for trade in closed if trade.net_pnl < 0]

            gross_profit = sum(trade.net_pnl for trade in wins)
            gross_loss = abs(sum(trade.net_pnl for trade in losses))
            net_profit = sum(trade.net_pnl for trade in closed)

            r_values = [
                trade.r_multiple for trade in closed
                if trade.r_multiple is not None
            ]

            stats = RegimePerformanceStats(
                regime=regime,
                trades=len(closed),
                net_profit=net_profit,
                win_rate=safe_div(len(wins), len(closed)) * 100.0,
                profit_factor=safe_div(gross_profit, gross_loss),
                max_drawdown_pct=self._estimate_trade_sequence_drawdown_pct(closed),
                average_r=mean(r_values) if r_values else None,
                metadata={
                    "gross_profit": gross_profit,
                    "gross_loss": gross_loss,
                    "enough_trades": len(closed) >= self.config.min_trades_for_regime_stats,
                },
            )
            result.append(stats)

        result.sort(key=lambda item: item.net_profit, reverse=True)
        return result

    @staticmethod
    def _signal_regime_map(signals: list[BacktestSignalRecord]) -> dict[str, str]:
        mapping: dict[str, str] = {}

        for signal in signals:
            if not signal.signal_id:
                continue

            regime = None

            if signal.metadata:
                regime = (
                    signal.metadata.get("market_regime")
                    or signal.metadata.get("regime")
                    or signal.metadata.get("context_regime")
                )

            if regime is None and isinstance(signal.payload, dict):
                regime = (
                    signal.payload.get("market_regime")
                    or signal.payload.get("regime")
                )

                context = signal.payload.get("context")
                if regime is None and isinstance(context, dict):
                    regime = context.get("market_regime") or context.get("regime")

            if regime is not None:
                mapping[signal.signal_id] = str(regime)

        return mapping

    @staticmethod
    def _extract_trade_regime(trade: SimulatedTrade) -> str | None:
        if not trade.metadata:
            return None

        regime = (
            trade.metadata.get("market_regime")
            or trade.metadata.get("regime")
            or trade.metadata.get("context_regime")
        )

        return str(regime) if regime is not None else None

    @staticmethod
    def _estimate_trade_sequence_drawdown_pct(trades: list[SimulatedTrade]) -> float:
        equity = 1.0
        peak = 1.0
        max_drawdown_pct = 0.0

        for trade in sorted(trades, key=lambda item: item.closed_at_ms or 0):
            equity += trade.net_pnl

            if equity > peak:
                peak = equity

            drawdown = max(0.0, peak - equity)
            drawdown_pct = safe_div(drawdown, peak) * 100.0 if peak > 0 else 0.0
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        return max_drawdown_pct


class FeatureImportanceAnalyzer:
    """
    Lightweight feature diagnostics.

    This is not ML feature importance. It analyzes which source features /
    confirmations / metadata keys appear in profitable vs unprofitable trades.
    """

    def __init__(self, config: ModelAnalyticsConfig | None = None) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

    def analyze(
        self,
        *,
        signals: list[BacktestSignalRecord],
        trades: list[SimulatedTrade],
    ) -> dict[str, Any]:
        signal_by_id = {
            signal.signal_id: signal
            for signal in signals
            if signal.signal_id
        }

        profitable_counter: Counter[str] = Counter()
        unprofitable_counter: Counter[str] = Counter()
        all_counter: Counter[str] = Counter()

        for trade in trades:
            if not trade.signal_id:
                continue

            signal = signal_by_id.get(trade.signal_id)
            if signal is None:
                continue

            features = self._extract_features(signal)

            for feature in features:
                all_counter[feature] += 1

                if trade.net_pnl > 0:
                    profitable_counter[feature] += 1
                elif trade.net_pnl < 0:
                    unprofitable_counter[feature] += 1

        feature_stats: dict[str, dict[str, Any]] = {}

        for feature, total in all_counter.items():
            profitable = profitable_counter[feature]
            unprofitable = unprofitable_counter[feature]

            feature_stats[feature] = {
                "total": total,
                "profitable": profitable,
                "unprofitable": unprofitable,
                "profitable_rate": safe_div(profitable, total) * 100.0,
                "unprofitable_rate": safe_div(unprofitable, total) * 100.0,
                "edge_score": safe_div(profitable - unprofitable, total),
            }

        ranked = sorted(
            feature_stats.items(),
            key=lambda item: item[1]["edge_score"],
            reverse=True,
        )

        return {
            "features": feature_stats,
            "top_positive": dict(ranked[:20]),
            "top_negative": dict(sorted(ranked, key=lambda item: item[1]["edge_score"])[:20]),
            "total_unique_features": len(feature_stats),
        }

    @staticmethod
    def _extract_features(signal: BacktestSignalRecord) -> set[str]:
        features: set[str] = set()

        for field_name in ("setup_type", "timeframe", "side"):
            value = getattr(signal, field_name, None)
            if value is not None:
                features.add(f"{field_name}:{value}")

        payload = signal.payload if isinstance(signal.payload, dict) else {}
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}

        for container_name, container in (("payload", payload), ("metadata", metadata)):
            for key in (
                "source_features",
                "features",
                "confirmations",
                "reasons",
                "tags",
            ):
                value = container.get(key)

                if isinstance(value, dict):
                    for item_key, item_value in value.items():
                        if isinstance(item_value, bool):
                            if item_value:
                                features.add(f"{key}:{item_key}")
                        else:
                            features.add(f"{key}:{item_key}")

                elif isinstance(value, list | tuple | set):
                    for item in value:
                        features.add(f"{key}:{item}")

                elif isinstance(value, str):
                    features.add(f"{key}:{value}")

            for key in (
                "market_regime",
                "regime",
                "category",
                "trigger_type",
                "priority",
                "origin",
            ):
                value = container.get(key)
                if value is not None:
                    features.add(f"{container_name}.{key}:{value}")

        return features


class RiskDecisionAnalyzer:
    """
    Analyzes RiskManager approvals/blocks.
    """

    def __init__(self, config: ModelAnalyticsConfig | None = None) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

    def analyze(
        self,
        *,
        risk_decisions: list[BacktestRiskDecisionRecord],
        trades: list[SimulatedTrade],
    ) -> RiskDecisionAnalytics:
        analytics = RiskDecisionAnalytics()

        if not risk_decisions:
            return analytics

        analytics.total_decisions = len(risk_decisions)
        analytics.approved = len([item for item in risk_decisions if item.approved])
        analytics.blocked = len([item for item in risk_decisions if item.blocked or not item.approved])
        analytics.approval_rate = safe_div(analytics.approved, analytics.total_decisions) * 100.0
        analytics.block_rate = safe_div(analytics.blocked, analytics.total_decisions) * 100.0

        reason_counter: Counter[str] = Counter()

        for decision in risk_decisions:
            if decision.blocked or not decision.approved:
                reason = str(decision.reason or "unknown")
                reason_counter[reason] += 1

        analytics.blocked_reason_counts = dict(reason_counter)

        approved_signal_ids = {
            decision.signal_id
            for decision in risk_decisions
            if decision.approved and decision.signal_id
        }

        analytics.approved_signals_pnl = sum(
            trade.net_pnl
            for trade in trades
            if trade.signal_id in approved_signal_ids
        )

        analytics.blocked_signals_count = analytics.blocked

        sizes = [item.final_size for item in risk_decisions if item.final_size is not None]
        leverages = [item.final_leverage for item in risk_decisions if item.final_leverage is not None]
        margins = [item.final_margin for item in risk_decisions if item.final_margin is not None]
        notionals = [item.final_notional for item in risk_decisions if item.final_notional is not None]

        analytics.avg_final_size = mean(sizes) if sizes else 0.0
        analytics.avg_final_leverage = mean(leverages) if leverages else 0.0
        analytics.avg_final_margin = mean(margins) if margins else 0.0
        analytics.avg_final_notional = mean(notionals) if notionals else 0.0

        analytics.max_final_size = max(sizes, default=0.0)
        analytics.max_final_leverage = max(leverages, default=0.0)
        analytics.max_final_margin = max(margins, default=0.0)
        analytics.max_final_notional = max(notionals, default=0.0)

        return analytics


class ExecutionQualityAnalyzer:
    """
    Analyzes simulated execution quality.
    """

    def __init__(self, config: ModelAnalyticsConfig | None = None) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

    def analyze(
        self,
        *,
        orders: list[SimulatedOrder],
        fills: list[SimulatedFill],
        execution_records: list[BacktestExecutionRecord],
    ) -> ExecutionQualityAnalytics:
        analytics = ExecutionQualityAnalytics()

        analytics.orders = len(orders)
        analytics.fills = len(fills)

        analytics.rejected_orders = len([
            order for order in orders
            if order.status.value == "rejected"
        ])
        analytics.cancelled_orders = len([
            order for order in orders
            if order.status.value == "cancelled"
        ])

        filled_orders = len([
            order for order in orders
            if order.status.value == "filled"
        ])

        analytics.fill_rate = safe_div(filled_orders, analytics.orders) * 100.0
        analytics.rejection_rate = safe_div(analytics.rejected_orders, analytics.orders) * 100.0
        analytics.cancellation_rate = safe_div(analytics.cancelled_orders, analytics.orders) * 100.0

        analytics.avg_slippage = mean([fill.slippage for fill in fills]) if fills else 0.0
        analytics.avg_slippage_bps = mean([fill.slippage_bps for fill in fills]) if fills else 0.0
        analytics.max_slippage_bps = max([fill.slippage_bps for fill in fills], default=0.0)

        analytics.total_fees = sum(fill.fee for fill in fills)
        analytics.total_slippage = sum(fill.slippage for fill in fills)

        latencies = [order.latency_ms for order in orders if order.latency_ms is not None]
        analytics.avg_latency_ms = mean(latencies) if latencies else 0.0

        rejection_counter: Counter[str] = Counter()

        for order in orders:
            if order.status.value == "rejected":
                rejection_counter[order.rejection_reason.value] += 1

        analytics.rejected_reason_counts = dict(rejection_counter)

        analytics.by_strategy = self._group_execution_by_strategy(orders=orders, fills=fills)
        analytics.by_symbol = self._group_execution_by_symbol(orders=orders, fills=fills)

        if not orders and execution_records:
            analytics.metadata["records_only_fallback"] = True
            analytics.orders = len(execution_records)

        return analytics

    @staticmethod
    def _group_execution_by_strategy(
        *,
        orders: list[SimulatedOrder],
        fills: list[SimulatedFill],
    ) -> dict[str, dict[str, Any]]:
        orders_by_strategy: dict[str, list[SimulatedOrder]] = defaultdict(list)
        fills_by_strategy: dict[str, list[SimulatedFill]] = defaultdict(list)

        for order in orders:
            strategy = order.strategy_name or "unknown"
            orders_by_strategy[strategy].append(order)

        for fill in fills:
            strategy = str(fill.metadata.get("strategy_name") or "unknown")
            fills_by_strategy[strategy].append(fill)

        result: dict[str, dict[str, Any]] = {}

        for strategy in sorted(set(orders_by_strategy) | set(fills_by_strategy)):
            strategy_orders = orders_by_strategy.get(strategy, [])
            strategy_fills = fills_by_strategy.get(strategy, [])

            result[strategy] = {
                "orders": len(strategy_orders),
                "fills": len(strategy_fills),
                "filled_orders": len([
                    order for order in strategy_orders
                    if order.status.value == "filled"
                ]),
                "rejected_orders": len([
                    order for order in strategy_orders
                    if order.status.value == "rejected"
                ]),
                "avg_slippage_bps": mean([fill.slippage_bps for fill in strategy_fills])
                if strategy_fills else 0.0,
                "total_fees": sum(fill.fee for fill in strategy_fills),
                "total_slippage": sum(fill.slippage for fill in strategy_fills),
            }

        return result

    @staticmethod
    def _group_execution_by_symbol(
        *,
        orders: list[SimulatedOrder],
        fills: list[SimulatedFill],
    ) -> dict[str, dict[str, Any]]:
        orders_by_symbol: dict[str, list[SimulatedOrder]] = defaultdict(list)
        fills_by_symbol: dict[str, list[SimulatedFill]] = defaultdict(list)

        for order in orders:
            orders_by_symbol[order.symbol].append(order)

        for fill in fills:
            fills_by_symbol[fill.symbol].append(fill)

        result: dict[str, dict[str, Any]] = {}

        for symbol in sorted(set(orders_by_symbol) | set(fills_by_symbol)):
            symbol_orders = orders_by_symbol.get(symbol, [])
            symbol_fills = fills_by_symbol.get(symbol, [])

            result[symbol] = {
                "orders": len(symbol_orders),
                "fills": len(symbol_fills),
                "filled_orders": len([
                    order for order in symbol_orders
                    if order.status.value == "filled"
                ]),
                "rejected_orders": len([
                    order for order in symbol_orders
                    if order.status.value == "rejected"
                ]),
                "avg_slippage_bps": mean([fill.slippage_bps for fill in symbol_fills])
                if symbol_fills else 0.0,
                "total_fees": sum(fill.fee for fill in symbol_fills),
                "total_slippage": sum(fill.slippage for fill in symbol_fills),
            }

        return result


class BacktestModelAnalyticsEngine:
    """
    Facade for all model analytics.
    """

    def __init__(
        self,
        config: ModelAnalyticsConfig | None = None,
    ) -> None:
        self.config = config or ModelAnalyticsConfig()
        self.config.validate()

        self.signal_quality = SignalQualityAnalyzer(self.config)
        self.strategy_attribution = StrategyAttributionAnalyzer(self.config)
        self.regime_performance = RegimePerformanceAnalyzer(self.config)
        self.feature_importance = FeatureImportanceAnalyzer(self.config)
        self.risk_decisions = RiskDecisionAnalyzer(self.config)
        self.execution_quality = ExecutionQualityAnalyzer(self.config)

    def analyze(
        self,
        data: ModelAnalyticsInput,
    ) -> BacktestModelAnalytics:
        """
        Build full model analytics result.
        """

        warnings: list[str] = []

        signal_quality = SignalQualityStats()
        strategy_attribution: list[StrategyAttribution] = []
        regime_performance: list[RegimePerformanceStats] = []
        feature_stats: dict[str, Any] = {}
        risk_decision_stats: dict[str, Any] = {}
        execution_quality_stats: dict[str, Any] = {}

        if self.config.analyze_signal_quality:
            try:
                signal_quality = self.signal_quality.analyze(
                    signals=data.signals,
                    trades=data.trades,
                    risk_decisions=data.risk_decisions,
                )
            except Exception as exc:
                warnings.append(f"signal_quality_failed: {exc}")

        if self.config.analyze_strategy_attribution:
            try:
                strategy_attribution = self.strategy_attribution.analyze(
                    trades=data.trades,
                    signals=data.signals,
                    risk_decisions=data.risk_decisions,
                )
            except Exception as exc:
                warnings.append(f"strategy_attribution_failed: {exc}")

        if self.config.analyze_regime_performance:
            try:
                regime_performance = self.regime_performance.analyze(
                    trades=data.trades,
                    signals=data.signals,
                )
            except Exception as exc:
                warnings.append(f"regime_performance_failed: {exc}")

        if self.config.analyze_feature_importance:
            try:
                feature_stats = self.feature_importance.analyze(
                    signals=data.signals,
                    trades=data.trades,
                )
            except Exception as exc:
                warnings.append(f"feature_importance_failed: {exc}")

        if self.config.analyze_risk_decisions:
            try:
                risk_decision_stats = self.risk_decisions.analyze(
                    risk_decisions=data.risk_decisions,
                    trades=data.trades,
                ).to_dict()
            except Exception as exc:
                warnings.append(f"risk_decisions_failed: {exc}")

        if self.config.analyze_execution_quality:
            try:
                execution_quality_stats = self.execution_quality.analyze(
                    orders=data.orders,
                    fills=data.fills,
                    execution_records=data.execution_records,
                ).to_dict()
            except Exception as exc:
                warnings.append(f"execution_quality_failed: {exc}")

        return BacktestModelAnalytics(
            signal_quality=signal_quality,
            strategy_attribution=strategy_attribution,
            regime_performance=regime_performance,
            feature_stats=feature_stats,
            risk_decision_stats=risk_decision_stats,
            execution_quality_stats=execution_quality_stats,
            warnings=warnings,
        )

    def analyze_from_components(
        self,
        *,
        signals: list[BacktestSignalRecord] | None = None,
        risk_decisions: list[BacktestRiskDecisionRecord] | None = None,
        orders: list[SimulatedOrder] | None = None,
        fills: list[SimulatedFill] | None = None,
        positions: list[SimulatedPosition] | None = None,
        trades: list[SimulatedTrade] | None = None,
        execution_records: list[BacktestExecutionRecord] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BacktestModelAnalytics:
        """
        Convenience wrapper around analyze().
        """

        return self.analyze(
            ModelAnalyticsInput(
                signals=signals or [],
                risk_decisions=risk_decisions or [],
                orders=orders or [],
                fills=fills or [],
                positions=positions or [],
                trades=trades or [],
                execution_records=execution_records or [],
                metadata=metadata or {},
            )
        )


# Backward-compatible short alias.
ModelAnalytics = BacktestModelAnalyticsEngine


# ============================================================================
# Standalone helpers
# ============================================================================


def rank_strategies_by_profit(
    attributions: Iterable[StrategyAttribution],
) -> list[StrategyAttribution]:
    """
    Rank strategy attribution records by net profit.
    """

    return sorted(
        attributions,
        key=lambda item: item.net_profit,
        reverse=True,
    )


def rank_strategies_by_profit_factor(
    attributions: Iterable[StrategyAttribution],
) -> list[StrategyAttribution]:
    """
    Rank strategy attribution records by profit factor.
    """

    return sorted(
        attributions,
        key=lambda item: item.profit_factor,
        reverse=True,
    )


def find_underperforming_strategies(
    attributions: Iterable[StrategyAttribution],
    *,
    min_trades: int = 10,
    min_profit_factor: float = 1.0,
    min_expectancy: float = 0.0,
) -> list[StrategyAttribution]:
    """
    Find strategies that appear weak after backtest.
    """

    result: list[StrategyAttribution] = []

    for item in attributions:
        if item.trades < min_trades:
            continue

        if item.profit_factor < min_profit_factor or item.expectancy < min_expectancy:
            result.append(item)

    return result


def summarize_model_analytics(analytics: BacktestModelAnalytics) -> dict[str, Any]:
    """
    Build compact analytics summary.
    """

    best_strategy = analytics.strategy_attribution[0] if analytics.strategy_attribution else None
    worst_strategy = analytics.strategy_attribution[-1] if analytics.strategy_attribution else None

    return {
        "signals_generated": analytics.signal_quality.signals_generated,
        "signals_confirmed": analytics.signal_quality.signals_confirmed,
        "signals_blocked_by_risk": analytics.signal_quality.signals_blocked_by_risk,
        "signals_executed": analytics.signal_quality.signals_executed,
        "profitable_signal_rate": analytics.signal_quality.profitable_signal_rate,
        "average_signal_pnl": analytics.signal_quality.average_signal_pnl,
        "best_strategy": best_strategy.strategy_name if best_strategy else None,
        "best_strategy_net_profit": best_strategy.net_profit if best_strategy else 0.0,
        "worst_strategy": worst_strategy.strategy_name if worst_strategy else None,
        "worst_strategy_net_profit": worst_strategy.net_profit if worst_strategy else 0.0,
        "regimes_analyzed": len(analytics.regime_performance),
        "features_analyzed": analytics.feature_stats.get("total_unique_features", 0)
        if isinstance(analytics.feature_stats, dict)
        else 0,
        "warnings": list(analytics.warnings),
    }


__all__ = [
    "ModelAnalyticsInput",
    "SignalAttributionRecord",
    "RiskDecisionAnalytics",
    "ExecutionQualityAnalytics",
    "SignalQualityAnalyzer",
    "StrategyAttributionAnalyzer",
    "RegimePerformanceAnalyzer",
    "FeatureImportanceAnalyzer",
    "RiskDecisionAnalyzer",
    "ExecutionQualityAnalyzer",
    "BacktestModelAnalyticsEngine",
    "ModelAnalytics",
    "rank_strategies_by_profit",
    "rank_strategies_by_profit_factor",
    "find_underperforming_strategies",
    "summarize_model_analytics",
]