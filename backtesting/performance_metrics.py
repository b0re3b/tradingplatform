"""
Performance metrics for backtesting.

This module calculates system, portfolio, strategy and symbol performance from
already collected backtest artifacts:

- simulated trades;
- simulated positions;
- equity curve;
- signal records;
- risk decision records;
- execution records;
- fills/orders;
- cost breakdowns.

Important:
- No EventBus usage here.
- No strategy/risk/execution decisions here.
- No live exchange calls here.
- This module is a pure analytics/calculation layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Iterable

from backtesting.config import PerformanceMetricsConfig
from backtesting.enums import MetricAggregation, SimulatedOrderStatus, SignalOutcome
from backtesting.exceptions import (
    DrawdownCalculationError,
    MetricInputError,
    RatioCalculationError,
    TradeStatsCalculationError,
)
from backtesting.models import (
    BacktestExecutionRecord,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    DrawdownPeriod,
    ExecutionStatsSnapshot,
    PerformanceSummary,
    PortfolioBacktestResult,
    RiskStats,
    SimulatedEquityPoint,
    SimulatedFill,
    SimulatedOrder,
    SimulatedPosition,
    SimulatedTrade,
    StrategyBacktestResult,
    SymbolBacktestResult,
    TradeStats,
    TradingCostBreakdown,
    safe_div,
)


@dataclass(slots=True)
class MetricsInput:
    """
    Input bundle for performance calculations.
    """

    initial_balance: float
    final_balance: float | None = None
    final_equity: float | None = None

    trades: list[SimulatedTrade] = field(default_factory=list)
    positions: list[SimulatedPosition] = field(default_factory=list)
    equity_curve: list[SimulatedEquityPoint] = field(default_factory=list)

    signals: list[BacktestSignalRecord] = field(default_factory=list)
    risk_decisions: list[BacktestRiskDecisionRecord] = field(default_factory=list)
    orders: list[SimulatedOrder] = field(default_factory=list)
    fills: list[SimulatedFill] = field(default_factory=list)
    execution_records: list[BacktestExecutionRecord] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.initial_balance <= 0:
            raise MetricInputError(
                "MetricsInput.initial_balance must be positive.",
                details={"initial_balance": self.initial_balance},
            )

        if self.final_balance is not None and self.final_balance < 0:
            raise MetricInputError(
                "MetricsInput.final_balance cannot be negative.",
                details={"final_balance": self.final_balance},
            )

        if self.final_equity is not None and self.final_equity < 0:
            raise MetricInputError(
                "MetricsInput.final_equity cannot be negative.",
                details={"final_equity": self.final_equity},
            )


@dataclass(slots=True)
class ReturnsSeries:
    """
    Equity return series used for ratio calculations.
    """

    returns: list[float] = field(default_factory=list)
    timestamps_ms: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.returns

    @property
    def count(self) -> int:
        return len(self.returns)


class TradeStatsCalculator:
    """
    Calculates trade statistics.
    """

    def calculate(self, trades: list[SimulatedTrade]) -> TradeStats:
        if trades is None:
            raise TradeStatsCalculationError("trades cannot be None.")

        stats = TradeStats()

        if not trades:
            return stats

        closed = [trade for trade in trades if trade.is_closed]
        open_trades = [trade for trade in trades if not trade.is_closed]

        wins = [trade for trade in closed if trade.net_pnl > 0]
        losses = [trade for trade in closed if trade.net_pnl < 0]
        breakeven = [trade for trade in closed if trade.net_pnl == 0]

        pnl_values = [trade.net_pnl for trade in closed]
        win_values = [trade.net_pnl for trade in wins]
        loss_values = [trade.net_pnl for trade in losses]

        gross_profit = sum(win_values)
        gross_loss = abs(sum(loss_values))
        net_profit = sum(pnl_values)

        stats.total_trades = len(trades)
        stats.open_trades = len(open_trades)
        stats.closed_trades = len(closed)
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.breakeven_trades = len(breakeven)

        stats.win_rate = safe_div(len(wins), len(closed)) * 100.0
        stats.loss_rate = safe_div(len(losses), len(closed)) * 100.0

        stats.gross_profit = gross_profit
        stats.gross_loss = gross_loss
        stats.net_profit = net_profit

        stats.average_trade = mean(pnl_values) if pnl_values else 0.0
        stats.average_win = mean(win_values) if win_values else 0.0
        stats.average_loss = mean(loss_values) if loss_values else 0.0
        stats.best_trade = max(pnl_values) if pnl_values else 0.0
        stats.worst_trade = min(pnl_values) if pnl_values else 0.0
        stats.median_trade = median(pnl_values) if pnl_values else 0.0

        stats.profit_factor = safe_div(gross_profit, gross_loss)
        stats.payoff_ratio = safe_div(stats.average_win, abs(stats.average_loss))

        if closed:
            stats.expectancy = (
                safe_div(len(wins), len(closed)) * stats.average_win
                - safe_div(len(losses), len(closed)) * abs(stats.average_loss)
            )

        r_values = [trade.r_multiple for trade in closed if trade.r_multiple is not None]
        stats.expectancy_r = mean(r_values) if r_values else None

        holding_times = [
            trade.holding_time_seconds
            for trade in closed
            if trade.holding_time_seconds > 0
        ]
        stats.average_holding_time_seconds = mean(holding_times) if holding_times else 0.0

        return stats


class DrawdownCalculator:
    """
    Calculates drawdown series and drawdown periods from equity curve.
    """

    def calculate_drawdowns(
        self,
        equity_curve: list[SimulatedEquityPoint],
        *,
        max_periods: int = 100,
    ) -> tuple[list[DrawdownPeriod], float, float, float]:
        """
        Returns:
            drawdown_periods, max_drawdown, max_drawdown_pct, average_drawdown
        """

        if equity_curve is None:
            raise DrawdownCalculationError("equity_curve cannot be None.")

        if not equity_curve:
            return [], 0.0, 0.0, 0.0

        points = sorted(equity_curve, key=lambda item: item.timestamp_ms)

        if any(point.equity < 0 for point in points):
            raise DrawdownCalculationError("Equity curve contains negative equity values.")

        peak_equity = points[0].equity
        peak_time = points[0].timestamp_ms
        valley_equity = points[0].equity
        valley_time = points[0].timestamp_ms

        in_drawdown = False
        current_start_ms = points[0].timestamp_ms
        current_peak_equity = peak_equity

        periods: list[DrawdownPeriod] = []
        drawdown_values: list[float] = []

        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        for point in points:
            equity = point.equity

            if equity >= peak_equity:
                if in_drawdown:
                    drawdown = current_peak_equity - valley_equity
                    drawdown_pct = safe_div(drawdown, current_peak_equity) * 100.0

                    periods.append(
                        DrawdownPeriod(
                            start_ms=current_start_ms,
                            valley_ms=valley_time,
                            recovery_ms=point.timestamp_ms,
                            peak_equity=current_peak_equity,
                            valley_equity=valley_equity,
                            drawdown=drawdown,
                            drawdown_pct=drawdown_pct,
                        )
                    )

                    in_drawdown = False

                peak_equity = equity
                peak_time = point.timestamp_ms
                valley_equity = equity
                valley_time = point.timestamp_ms
                continue

            drawdown = peak_equity - equity
            drawdown_pct = safe_div(drawdown, peak_equity) * 100.0 if peak_equity > 0 else 0.0

            drawdown_values.append(drawdown)

            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

            if not in_drawdown:
                in_drawdown = True
                current_start_ms = peak_time
                current_peak_equity = peak_equity
                valley_equity = equity
                valley_time = point.timestamp_ms
            elif equity < valley_equity:
                valley_equity = equity
                valley_time = point.timestamp_ms

        if in_drawdown:
            drawdown = current_peak_equity - valley_equity
            drawdown_pct = safe_div(drawdown, current_peak_equity) * 100.0

            periods.append(
                DrawdownPeriod(
                    start_ms=current_start_ms,
                    valley_ms=valley_time,
                    recovery_ms=None,
                    peak_equity=current_peak_equity,
                    valley_equity=valley_equity,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                )
            )

        periods.sort(key=lambda item: item.drawdown_pct, reverse=True)

        if max_periods > 0:
            periods = periods[:max_periods]

        average_drawdown = mean(drawdown_values) if drawdown_values else 0.0
        return periods, max_drawdown, max_drawdown_pct, average_drawdown


class RatioCalculator:
    """
    Calculates Sharpe, Sortino, Calmar and recovery factor.
    """

    def __init__(self, config: PerformanceMetricsConfig | None = None) -> None:
        self.config = config or PerformanceMetricsConfig()
        self.config.validate()

    def build_returns(
        self,
        equity_curve: list[SimulatedEquityPoint],
    ) -> ReturnsSeries:
        if equity_curve is None:
            raise RatioCalculationError("equity_curve cannot be None.")

        points = sorted(equity_curve, key=lambda item: item.timestamp_ms)

        if len(points) < 2:
            return ReturnsSeries()

        returns: list[float] = []
        timestamps: list[int] = []

        for previous, current in zip(points, points[1:]):
            if previous.equity <= 0:
                continue

            if self.config.use_log_returns:
                if current.equity <= 0:
                    continue
                value = math.log(current.equity / previous.equity)
            else:
                value = current.equity / previous.equity - 1.0

            returns.append(value)
            timestamps.append(current.timestamp_ms)

        return ReturnsSeries(returns=returns, timestamps_ms=timestamps)

    def sharpe_ratio(self, returns: ReturnsSeries) -> float | None:
        if returns.count < 2:
            return None

        excess = [
            item - self.config.risk_free_rate / self.config.annualization_periods
            for item in returns.returns
        ]

        deviation = pstdev(excess)

        if deviation == 0:
            return None

        return mean(excess) / deviation * math.sqrt(self.config.annualization_periods)

    def sortino_ratio(self, returns: ReturnsSeries) -> float | None:
        if returns.count < 2:
            return None

        target_return = self.config.risk_free_rate / self.config.annualization_periods
        excess = [item - target_return for item in returns.returns]
        downside = [min(0.0, item) for item in excess]

        if not downside:
            return None

        downside_deviation = math.sqrt(mean([item * item for item in downside]))

        if downside_deviation == 0:
            return None

        return mean(excess) / downside_deviation * math.sqrt(self.config.annualization_periods)

    @staticmethod
    def calmar_ratio(
        *,
        net_profit_pct: float,
        max_drawdown_pct: float,
        years: float | None = None,
    ) -> float | None:
        if max_drawdown_pct <= 0:
            return None

        annualized_return_pct = net_profit_pct

        if years is not None and years > 0:
            annualized_return_pct = net_profit_pct / years

        return annualized_return_pct / max_drawdown_pct

    @staticmethod
    def recovery_factor(
        *,
        net_profit: float,
        max_drawdown: float,
    ) -> float | None:
        if max_drawdown <= 0:
            return None

        return net_profit / max_drawdown


class RiskStatsCalculator:
    """
    Calculates risk pipeline stats from signal and risk decision records.
    """

    def calculate(
        self,
        *,
        signals: list[BacktestSignalRecord],
        risk_decisions: list[BacktestRiskDecisionRecord],
    ) -> RiskStats:
        stats = RiskStats()

        stats.signals_received = len(signals)
        stats.signals_confirmed = len([item for item in risk_decisions if item.approved])
        stats.signals_blocked = len([item for item in risk_decisions if item.blocked or not item.approved])

        if not risk_decisions:
            stats.signals_confirmed = len([
                signal
                for signal in signals
                if signal.outcome in {
                    SignalOutcome.CONFIRMED_BY_RISK,
                    SignalOutcome.ORDER_FILLED,
                    SignalOutcome.POSITION_OPENED,
                    SignalOutcome.POSITION_CLOSED_WIN,
                    SignalOutcome.POSITION_CLOSED_LOSS,
                    SignalOutcome.POSITION_CLOSED_BREAKEVEN,
                }
            ])
            stats.signals_blocked = len([
                signal
                for signal in signals
                if signal.outcome == SignalOutcome.BLOCKED_BY_RISK
            ])

        stats.confirmation_rate = safe_div(stats.signals_confirmed, stats.signals_received) * 100.0
        stats.block_rate = safe_div(stats.signals_blocked, stats.signals_received) * 100.0

        stats.position_blocked_events = stats.signals_blocked
        stats.kill_switch_events = len([
            item
            for item in risk_decisions
            if str(item.reason or "").lower() in {"kill_switch", "risk.kill_switch"}
        ])
        stats.limit_warnings = len([
            item
            for item in risk_decisions
            if "limit" in str(item.reason or "").lower()
        ])

        final_margins = [item.final_margin for item in risk_decisions if item.final_margin is not None]
        final_notional = [item.final_notional for item in risk_decisions if item.final_notional is not None]
        final_leverage = [item.final_leverage for item in risk_decisions if item.final_leverage is not None]

        stats.max_margin_used = max(final_margins, default=0.0)
        stats.max_exposure = max(final_notional, default=0.0)
        stats.max_leverage_used = max(final_leverage, default=0.0)

        reservations = [item for item in risk_decisions if item.reservation_id]
        stats.reservations_created = len(reservations)

        # Released/expired reservations are usually inferred from execution/risk events
        # later in StrategyTester. Keep zero here unless specific metadata is present.
        stats.reservations_released = len([
            item
            for item in risk_decisions
            if item.metadata.get("reservation_released") is True
        ])
        stats.reservations_expired = len([
            item
            for item in risk_decisions
            if item.metadata.get("reservation_expired") is True
        ])

        return stats


class ExecutionStatsCalculator:
    """
    Calculates simulated execution stats.
    """

    def calculate(
        self,
        *,
        orders: list[SimulatedOrder],
        fills: list[SimulatedFill],
        execution_records: list[BacktestExecutionRecord],
    ) -> ExecutionStatsSnapshot:
        stats = ExecutionStatsSnapshot()

        stats.orders_submitted = len([
            order
            for order in orders
            if order.submitted_at_ms is not None or order.status != SimulatedOrderStatus.CREATED
        ])
        stats.orders_accepted = len([
            order
            for order in orders
            if order.accepted_at_ms is not None
        ])
        stats.orders_rejected = len([
            order
            for order in orders
            if order.status == SimulatedOrderStatus.REJECTED
        ])
        stats.orders_cancelled = len([
            order
            for order in orders
            if order.status == SimulatedOrderStatus.CANCELLED
        ])
        stats.orders_filled = len([
            order
            for order in orders
            if order.status == SimulatedOrderStatus.FILLED
        ])
        stats.orders_partially_filled = len([
            order
            for order in orders
            if order.status == SimulatedOrderStatus.PARTIALLY_FILLED
        ])
        stats.fills = len(fills)

        terminal_orders = len([
            order
            for order in orders
            if order.is_terminal
        ])

        stats.rejection_rate = safe_div(stats.orders_rejected, stats.orders_submitted) * 100.0
        stats.fill_rate = safe_div(stats.orders_filled, terminal_orders) * 100.0
        stats.partial_fill_rate = safe_div(stats.orders_partially_filled, stats.orders_submitted) * 100.0

        slippages = [fill.slippage for fill in fills]
        slippage_bps = [fill.slippage_bps for fill in fills]
        latencies = [order.latency_ms for order in orders if order.latency_ms is not None]

        stats.average_slippage = mean(slippages) if slippages else 0.0
        stats.average_slippage_bps = mean(slippage_bps) if slippage_bps else 0.0
        stats.average_latency_ms = mean(latencies) if latencies else 0.0

        stats.total_fees = sum(fill.fee for fill in fills)
        stats.total_slippage = sum(fill.slippage for fill in fills)

        # If only records exist and orders/fills were not persisted, use records
        # as a fallback for rough order counters.
        if not orders and execution_records:
            stats.orders_submitted = len([
                record
                for record in execution_records
                if record.topic.endswith("order_submitted")
            ])
            stats.orders_rejected = len([
                record
                for record in execution_records
                if record.topic.endswith("order_rejected")
            ])
            stats.orders_cancelled = len([
                record
                for record in execution_records
                if record.topic.endswith("order_cancelled")
            ])
            stats.orders_filled = len([
                record
                for record in execution_records
                if record.topic.endswith("order_filled")
            ])
            stats.orders_partially_filled = len([
                record
                for record in execution_records
                if record.topic.endswith("order_partially_filled")
            ])

        return stats


class CostStatsCalculator:
    """
    Aggregates trading costs from trades, positions and fills.
    """

    def calculate(
        self,
        *,
        trades: list[SimulatedTrade],
        positions: list[SimulatedPosition],
        fills: list[SimulatedFill],
    ) -> TradingCostBreakdown:
        breakdown = TradingCostBreakdown()

        if fills:
            breakdown.commission = sum(fill.fee for fill in fills)
            breakdown.slippage = sum(fill.slippage for fill in fills)

        if positions:
            breakdown.funding_paid = sum(position.funding_paid for position in positions)
            breakdown.funding_received = sum(position.funding_received for position in positions)

            if not fills:
                breakdown.commission = sum(position.fees_paid for position in positions)
                breakdown.slippage = sum(position.slippage_paid for position in positions)

        if trades and not positions and not fills:
            breakdown.commission = sum(trade.fees for trade in trades)
            breakdown.slippage = sum(trade.slippage for trade in trades)

            funding = sum(trade.funding for trade in trades)
            if funding >= 0:
                breakdown.funding_received = funding
            else:
                breakdown.funding_paid = abs(funding)

        return breakdown


class PerformanceMetrics:
    """
    Main facade for calculating backtest performance metrics.
    """

    def __init__(
        self,
        config: PerformanceMetricsConfig | None = None,
    ) -> None:
        self.config = config or PerformanceMetricsConfig()
        self.config.validate()

        self.trade_stats_calculator = TradeStatsCalculator()
        self.drawdown_calculator = DrawdownCalculator()
        self.ratio_calculator = RatioCalculator(self.config)
        self.risk_stats_calculator = RiskStatsCalculator()
        self.execution_stats_calculator = ExecutionStatsCalculator()
        self.cost_stats_calculator = CostStatsCalculator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_portfolio_result(
        self,
        data: MetricsInput,
    ) -> PortfolioBacktestResult:
        """
        Calculate full portfolio-level backtest result.
        """

        data.validate()

        summary = self.calculate_summary(
            data,
            aggregation=MetricAggregation.SYSTEM,
            key="system",
        )

        trade_stats = (
            self.trade_stats_calculator.calculate(data.trades)
            if self.config.calculate_trade_stats
            else TradeStats()
        )

        risk_stats = (
            self.risk_stats_calculator.calculate(
                signals=data.signals,
                risk_decisions=data.risk_decisions,
            )
            if self.config.calculate_risk_stats
            else RiskStats()
        )

        execution_stats = (
            self.execution_stats_calculator.calculate(
                orders=data.orders,
                fills=data.fills,
                execution_records=data.execution_records,
            )
            if self.config.calculate_execution_stats
            else ExecutionStatsSnapshot()
        )

        drawdowns: list[DrawdownPeriod] = []
        costs = TradingCostBreakdown()

        if self.config.calculate_drawdowns:
            drawdowns, _, _, _ = self.drawdown_calculator.calculate_drawdowns(
                data.equity_curve,
                max_periods=self.config.max_drawdown_periods,
            )

        if self.config.calculate_cost_breakdown:
            costs = self.cost_stats_calculator.calculate(
                trades=data.trades,
                positions=data.positions,
                fills=data.fills,
            )

        strategy_results = (
            self.calculate_strategy_results(data)
            if self.config.calculate_strategy_breakdown
            else {}
        )

        symbol_results = (
            self.calculate_symbol_results(data)
            if self.config.calculate_symbol_breakdown
            else {}
        )

        return PortfolioBacktestResult(
            summary=summary,
            trade_stats=trade_stats,
            risk_stats=risk_stats,
            execution_stats=execution_stats,
            strategy_results=strategy_results,
            symbol_results=symbol_results,
            equity_curve=list(data.equity_curve),
            drawdowns=drawdowns,
            costs=costs,
            metadata=dict(data.metadata),
        )

    def calculate_summary(
        self,
        data: MetricsInput,
        *,
        aggregation: MetricAggregation = MetricAggregation.SYSTEM,
        key: str = "system",
    ) -> PerformanceSummary:
        """
        Calculate one summary block.
        """

        data.validate()

        final_equity = self._resolve_final_equity(data)
        final_balance = self._resolve_final_balance(data, final_equity)

        net_profit = final_equity - data.initial_balance
        net_profit_pct = safe_div(net_profit, data.initial_balance) * 100.0

        trade_stats = self.trade_stats_calculator.calculate(data.trades)
        cost_breakdown = self.cost_stats_calculator.calculate(
            trades=data.trades,
            positions=data.positions,
            fills=data.fills,
        )

        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        average_drawdown = 0.0
        drawdowns: list[DrawdownPeriod] = []

        if self.config.calculate_drawdowns and data.equity_curve:
            drawdowns, max_drawdown, max_drawdown_pct, average_drawdown = (
                self.drawdown_calculator.calculate_drawdowns(
                    data.equity_curve,
                    max_periods=self.config.max_drawdown_periods,
                )
            )

        returns = self.ratio_calculator.build_returns(data.equity_curve)
        sharpe_ratio = None
        sortino_ratio = None

        if self.config.calculate_ratios and returns.count >= self.config.min_trades_for_ratios:
            sharpe_ratio = self.ratio_calculator.sharpe_ratio(returns)
            sortino_ratio = self.ratio_calculator.sortino_ratio(returns)

        years = self._estimate_years(data.equity_curve)
        calmar_ratio = self.ratio_calculator.calmar_ratio(
            net_profit_pct=net_profit_pct,
            max_drawdown_pct=max_drawdown_pct,
            years=years,
        )
        recovery_factor = self.ratio_calculator.recovery_factor(
            net_profit=net_profit,
            max_drawdown=max_drawdown,
        )

        exposure_time_pct = self._calculate_exposure_time_pct(
            trades=data.trades,
            equity_curve=data.equity_curve,
        )

        return PerformanceSummary(
            aggregation=aggregation,
            key=key,
            initial_balance=data.initial_balance,
            final_balance=final_balance,
            final_equity=final_equity,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            gross_profit=trade_stats.gross_profit,
            gross_loss=trade_stats.gross_loss,
            profit_factor=trade_stats.profit_factor,
            expectancy=trade_stats.expectancy,
            total_trades=trade_stats.closed_trades,
            win_rate=trade_stats.win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            average_drawdown=average_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
            exposure_time_pct=exposure_time_pct,
            total_fees=cost_breakdown.commission,
            total_slippage=cost_breakdown.slippage,
            total_funding=cost_breakdown.net_funding,
            metadata={
                "drawdown_periods": len(drawdowns),
                "returns_count": returns.count,
                "years": years,
            },
        )

    def calculate_strategy_results(
        self,
        data: MetricsInput,
    ) -> dict[str, StrategyBacktestResult]:
        """
        Calculate results grouped by strategy_name.
        """

        strategy_names = sorted(
            {
                trade.strategy_name
                for trade in data.trades
                if trade.strategy_name
            }
            | {
                signal.strategy_name
                for signal in data.signals
                if signal.strategy_name
            }
            | {
                position.strategy_name
                for position in data.positions
                if position.strategy_name
            }
        )

        results: dict[str, StrategyBacktestResult] = {}

        for strategy_name in strategy_names:
            trades = [trade for trade in data.trades if trade.strategy_name == strategy_name]
            positions = [position for position in data.positions if position.strategy_name == strategy_name]
            signals = [signal for signal in data.signals if signal.strategy_name == strategy_name]
            risk_decisions = [
                item
                for item in data.risk_decisions
                if item.strategy_name == strategy_name
            ]
            orders = [order for order in data.orders if order.strategy_name == strategy_name]
            fills = [
                fill
                for fill in data.fills
                if fill.metadata.get("strategy_name") == strategy_name
            ]
            execution_records = [
                record
                for record in data.execution_records
                if record.strategy_name == strategy_name
            ]

            strategy_equity = self._build_strategy_equity_curve(
                initial_balance=data.initial_balance,
                trades=trades,
                global_equity_curve=data.equity_curve,
            )

            strategy_input = MetricsInput(
                initial_balance=data.initial_balance,
                final_equity=(
                    strategy_equity[-1].equity
                    if strategy_equity
                    else data.initial_balance + sum(trade.net_pnl for trade in trades)
                ),
                trades=trades,
                positions=positions,
                equity_curve=strategy_equity,
                signals=signals,
                risk_decisions=risk_decisions,
                orders=orders,
                fills=fills,
                execution_records=execution_records,
                metadata={"strategy_name": strategy_name},
            )

            summary = self.calculate_summary(
                strategy_input,
                aggregation=MetricAggregation.STRATEGY,
                key=strategy_name,
            )
            trade_stats = self.trade_stats_calculator.calculate(trades)
            risk_stats = self.risk_stats_calculator.calculate(
                signals=signals,
                risk_decisions=risk_decisions,
            )
            execution_stats = self.execution_stats_calculator.calculate(
                orders=orders,
                fills=fills,
                execution_records=execution_records,
            )

            results[strategy_name] = StrategyBacktestResult(
                strategy_name=strategy_name,
                summary=summary,
                trade_stats=trade_stats,
                risk_stats=risk_stats,
                execution_stats=execution_stats,
                signals=signals,
                trades=trades,
                positions=positions,
                equity_curve=strategy_equity,
            )

        return results

    def calculate_symbol_results(
        self,
        data: MetricsInput,
    ) -> dict[str, SymbolBacktestResult]:
        """
        Calculate results grouped by symbol.
        """

        symbols = sorted(
            {
                trade.symbol
                for trade in data.trades
                if trade.symbol
            }
            | {
                position.symbol
                for position in data.positions
                if position.symbol
            }
        )

        results: dict[str, SymbolBacktestResult] = {}

        for symbol in symbols:
            trades = [trade for trade in data.trades if trade.symbol == symbol]
            positions = [position for position in data.positions if position.symbol == symbol]

            symbol_equity = self._build_strategy_equity_curve(
                initial_balance=data.initial_balance,
                trades=trades,
                global_equity_curve=data.equity_curve,
            )

            symbol_input = MetricsInput(
                initial_balance=data.initial_balance,
                final_equity=(
                    symbol_equity[-1].equity
                    if symbol_equity
                    else data.initial_balance + sum(trade.net_pnl for trade in trades)
                ),
                trades=trades,
                positions=positions,
                equity_curve=symbol_equity,
                metadata={"symbol": symbol},
            )

            summary = self.calculate_summary(
                symbol_input,
                aggregation=MetricAggregation.SYMBOL,
                key=symbol,
            )
            trade_stats = self.trade_stats_calculator.calculate(trades)

            exchange = positions[0].exchange if positions else (trades[0].exchange if trades else "binance")
            market_type = positions[0].market_type if positions else (trades[0].market_type if trades else "usdm_futures")

            results[symbol] = SymbolBacktestResult(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                summary=summary,
                trade_stats=trade_stats,
                trades=trades,
                positions=positions,
            )

        return results

    # ------------------------------------------------------------------
    # Convenience public helpers
    # ------------------------------------------------------------------

    def calculate_trade_stats(self, trades: list[SimulatedTrade]) -> TradeStats:
        return self.trade_stats_calculator.calculate(trades)

    def calculate_risk_stats(
        self,
        *,
        signals: list[BacktestSignalRecord],
        risk_decisions: list[BacktestRiskDecisionRecord],
    ) -> RiskStats:
        return self.risk_stats_calculator.calculate(
            signals=signals,
            risk_decisions=risk_decisions,
        )

    def calculate_execution_stats(
        self,
        *,
        orders: list[SimulatedOrder],
        fills: list[SimulatedFill],
        execution_records: list[BacktestExecutionRecord],
    ) -> ExecutionStatsSnapshot:
        return self.execution_stats_calculator.calculate(
            orders=orders,
            fills=fills,
            execution_records=execution_records,
        )

    def calculate_drawdowns(
        self,
        equity_curve: list[SimulatedEquityPoint],
    ) -> list[DrawdownPeriod]:
        drawdowns, _, _, _ = self.drawdown_calculator.calculate_drawdowns(
            equity_curve,
            max_periods=self.config.max_drawdown_periods,
        )
        return drawdowns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_final_equity(data: MetricsInput) -> float:
        if data.final_equity is not None:
            return data.final_equity

        if data.equity_curve:
            return sorted(data.equity_curve, key=lambda item: item.timestamp_ms)[-1].equity

        if data.trades:
            return data.initial_balance + sum(trade.net_pnl for trade in data.trades if trade.is_closed)

        return data.initial_balance

    @staticmethod
    def _resolve_final_balance(data: MetricsInput, final_equity: float) -> float:
        if data.final_balance is not None:
            return data.final_balance

        if data.equity_curve:
            return sorted(data.equity_curve, key=lambda item: item.timestamp_ms)[-1].balance

        return final_equity

    @staticmethod
    def _estimate_years(equity_curve: list[SimulatedEquityPoint]) -> float | None:
        if len(equity_curve) < 2:
            return None

        points = sorted(equity_curve, key=lambda item: item.timestamp_ms)
        start_ms = points[0].timestamp_ms
        end_ms = points[-1].timestamp_ms

        seconds = max(0.0, (end_ms - start_ms) / 1000.0)

        if seconds <= 0:
            return None

        return seconds / (365.0 * 24.0 * 60.0 * 60.0)

    @staticmethod
    def _calculate_exposure_time_pct(
        *,
        trades: list[SimulatedTrade],
        equity_curve: list[SimulatedEquityPoint],
    ) -> float:
        if not trades or len(equity_curve) < 2:
            return 0.0

        points = sorted(equity_curve, key=lambda item: item.timestamp_ms)
        total_period_ms = points[-1].timestamp_ms - points[0].timestamp_ms

        if total_period_ms <= 0:
            return 0.0

        exposed_ms = 0

        for trade in trades:
            if trade.opened_at_ms is None:
                continue

            end_ms = trade.closed_at_ms or points[-1].timestamp_ms
            exposed_ms += max(0, end_ms - trade.opened_at_ms)

        return min(100.0, exposed_ms / total_period_ms * 100.0)

    @staticmethod
    def _build_strategy_equity_curve(
        *,
        initial_balance: float,
        trades: list[SimulatedTrade],
        global_equity_curve: list[SimulatedEquityPoint],
    ) -> list[SimulatedEquityPoint]:
        """
        Build simple strategy/symbol equity curve from closed trade PnL.

        This is attribution-oriented, not a replacement for real portfolio
        accounting. It approximates curve by applying closed trade PnL at close
        timestamps.
        """

        if not trades:
            if global_equity_curve:
                first = sorted(global_equity_curve, key=lambda item: item.timestamp_ms)[0]
                return [
                    SimulatedEquityPoint(
                        timestamp_ms=first.timestamp_ms,
                        equity=initial_balance,
                        balance=initial_balance,
                        available_balance=initial_balance,
                        source="performance_metrics.synthetic_empty",
                    )
                ]
            return []

        closed_trades = sorted(
            [trade for trade in trades if trade.closed_at_ms is not None],
            key=lambda trade: trade.closed_at_ms or 0,
        )

        if not closed_trades:
            return []

        equity = initial_balance
        curve: list[SimulatedEquityPoint] = []

        first_timestamp = closed_trades[0].opened_at_ms or closed_trades[0].closed_at_ms or 0
        curve.append(
            SimulatedEquityPoint(
                timestamp_ms=first_timestamp,
                equity=equity,
                balance=equity,
                available_balance=equity,
                source="performance_metrics.synthetic_start",
            )
        )

        peak = equity

        for trade in closed_trades:
            equity += trade.net_pnl
            peak = max(peak, equity)
            drawdown = max(0.0, peak - equity)
            drawdown_pct = safe_div(drawdown, peak) * 100.0 if peak > 0 else 0.0

            curve.append(
                SimulatedEquityPoint(
                    timestamp_ms=trade.closed_at_ms or first_timestamp,
                    equity=equity,
                    balance=equity,
                    available_balance=equity,
                    realized_pnl=equity - initial_balance,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                    source="performance_metrics.synthetic_trade_close",
                    metadata={
                        "trade_id": trade.trade_id,
                        "strategy_name": trade.strategy_name,
                        "symbol": trade.symbol,
                    },
                )
            )

        return curve


# ============================================================================
# Standalone helpers
# ============================================================================


def calculate_profit_factor(trades: Iterable[SimulatedTrade]) -> float:
    """
    Calculate profit factor from trades.
    """

    closed = [trade for trade in trades if trade.is_closed]
    gross_profit = sum(trade.net_pnl for trade in closed if trade.net_pnl > 0)
    gross_loss = abs(sum(trade.net_pnl for trade in closed if trade.net_pnl < 0))
    return safe_div(gross_profit, gross_loss)


def calculate_win_rate(trades: Iterable[SimulatedTrade]) -> float:
    """
    Calculate win rate percentage from trades.
    """

    closed = [trade for trade in trades if trade.is_closed]

    if not closed:
        return 0.0

    wins = len([trade for trade in closed if trade.net_pnl > 0])
    return wins / len(closed) * 100.0


def calculate_expectancy(trades: Iterable[SimulatedTrade]) -> float:
    """
    Calculate expectancy from trades.
    """

    closed = [trade for trade in trades if trade.is_closed]

    if not closed:
        return 0.0

    wins = [trade.net_pnl for trade in closed if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in closed if trade.net_pnl < 0]

    win_rate = safe_div(len(wins), len(closed))
    loss_rate = safe_div(len(losses), len(closed))

    average_win = mean(wins) if wins else 0.0
    average_loss = abs(mean(losses)) if losses else 0.0

    return win_rate * average_win - loss_rate * average_loss


def calculate_max_drawdown_pct(equity_curve: list[SimulatedEquityPoint]) -> float:
    """
    Calculate max drawdown percentage from equity curve.
    """

    calculator = DrawdownCalculator()
    _, _, max_drawdown_pct, _ = calculator.calculate_drawdowns(equity_curve)
    return max_drawdown_pct


def build_metrics_input_from_components(
    *,
    initial_balance: float,
    final_balance: float | None = None,
    final_equity: float | None = None,
    trades: list[SimulatedTrade] | None = None,
    positions: list[SimulatedPosition] | None = None,
    equity_curve: list[SimulatedEquityPoint] | None = None,
    signals: list[BacktestSignalRecord] | None = None,
    risk_decisions: list[BacktestRiskDecisionRecord] | None = None,
    orders: list[SimulatedOrder] | None = None,
    fills: list[SimulatedFill] | None = None,
    execution_records: list[BacktestExecutionRecord] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MetricsInput:
    """
    Convenience builder for MetricsInput.
    """

    return MetricsInput(
        initial_balance=initial_balance,
        final_balance=final_balance,
        final_equity=final_equity,
        trades=trades or [],
        positions=positions or [],
        equity_curve=equity_curve or [],
        signals=signals or [],
        risk_decisions=risk_decisions or [],
        orders=orders or [],
        fills=fills or [],
        execution_records=execution_records or [],
        metadata=metadata or {},
    )


__all__ = [
    "MetricsInput",
    "ReturnsSeries",
    "TradeStatsCalculator",
    "DrawdownCalculator",
    "RatioCalculator",
    "RiskStatsCalculator",
    "ExecutionStatsCalculator",
    "CostStatsCalculator",
    "PerformanceMetrics",
    "calculate_profit_factor",
    "calculate_win_rate",
    "calculate_expectancy",
    "calculate_max_drawdown_pct",
    "build_metrics_input_from_components",
]