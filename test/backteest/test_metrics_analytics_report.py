"""
Tests for performance metrics, model analytics and report building.

Covered modules:
- backtesting.performance_metrics
- backtesting.model_analytics
- backtesting.report_builder

These tests focus on post-run correctness:
- portfolio / strategy / symbol metrics;
- risk and execution aggregates;
- signal quality, attribution, regime and feature analytics;
- report rendering and artifact exports.

The tests intentionally use small deterministic fixtures and compatibility
helpers so they stay focused on backtesting output behavior rather than on the
full live trading pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backtesting.config import (
    ModelAnalyticsConfig,
    PerformanceMetricsConfig,
    ReportBuilderConfig,
)
from backtesting.enums import (
    BacktestArtifactType,
    BacktestMode,
    BacktestStatus,
    MetricAggregation,
    OrderRejectionReason,
    ReportFormat,
    ReportSection,
    SignalOutcome,
    SimulatedOrderStatus,
    SimulatedPositionStatus,
    TradeOutcome,
)
from backtesting.exceptions import MetricInputError
from backtesting.model_analytics import (
    BacktestModelAnalyticsEngine,
    ModelAnalyticsInput,
    find_underperforming_strategies,
    rank_strategies_by_profit,
    rank_strategies_by_profit_factor,
    summarize_model_analytics,
)
from backtesting.models import (
    BacktestExecutionRecord,
    BacktestPeriod,
    BacktestResult,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    PerformanceSummary,
    SimulatedEquityPoint,
    SimulatedFill,
    SimulatedOrder,
    SimulatedPosition,
    SimulatedTrade,
    timestamp_ms,
)
from backtesting.performance_metrics import (
    CostStatsCalculator,
    DrawdownCalculator,
    ExecutionStatsCalculator,
    MetricsInput,
    PerformanceMetrics,
    RiskStatsCalculator,
    TradeStatsCalculator,
    build_metrics_input_from_components,
    calculate_expectancy,
    calculate_max_drawdown_pct,
    calculate_profit_factor,
    calculate_win_rate,
)
from backtesting.report_builder import ReportBuilder, build_markdown_report, export_backtest_report


# =============================================================================
# Compatibility helpers
# =============================================================================


def _closed_trade(trade: SimulatedTrade, outcome: TradeOutcome) -> SimulatedTrade:
    """
    Fixtures keep trades lightweight, so make closure explicit for metrics.
    """

    return trade.copy_with(outcome=outcome)


def _sample_trades(
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
) -> list[SimulatedTrade]:
    return [
        _closed_trade(sample_trade, TradeOutcome.WIN),
        _closed_trade(losing_trade, TradeOutcome.LOSS),
    ]


def _sample_orders() -> list[SimulatedOrder]:
    return [
        SimulatedOrder(
            order_id="order_1",
            run_id="test_run",
            signal_id="signal_1",
            strategy_name="fake_strategy",
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            side="buy",
            order_type="market",
            status=SimulatedOrderStatus.FILLED,
            quantity=1.0,
            filled_quantity=1.0,
            remaining_quantity=0.0,
            price=100.0,
            average_fill_price=100.0,
            submitted_at_ms=1,
            accepted_at_ms=1,
            filled_at_ms=2,
            fees=0.04,
            slippage=0.02,
            latency_ms=0,
            metadata={"source": "test"},
        ),
        SimulatedOrder(
            order_id="order_2",
            run_id="test_run",
            signal_id="signal_2",
            strategy_name="fake_strategy",
            exchange="binance",
            symbol="BTCUSDT",
            market_type="usdm_futures",
            side="buy",
            order_type="market",
            status=SimulatedOrderStatus.REJECTED,
            quantity=1.0,
            filled_quantity=0.0,
            remaining_quantity=1.0,
            price=100.0,
            rejected_at_ms=3,
            rejection_reason=OrderRejectionReason.RISK_REJECTED,
            rejection_message="blocked by test risk guard",
            metadata={"source": "test"},
        ),
    ]


def _sample_position() -> SimulatedPosition:
    return SimulatedPosition(
        position_id="position_1",
        run_id="test_run",
        signal_id="signal_1",
        strategy_name="fake_strategy",
        exchange="binance",
        symbol="BTCUSDT",
        market_type="usdm_futures",
        side="long",
        status=SimulatedPositionStatus.CLOSED,
        quantity=1.0,
        entry_price=100.0,
        mark_price=110.0,
        exit_price=110.0,
        leverage=2.0,
        realized_pnl=9.9,
        unrealized_pnl=0.0,
        fees_paid=0.08,
        slippage_paid=0.02,
        opened_at_ms=1,
        closed_at_ms=2,
        close_reason="take_profit",
        source_order_ids=["order_1"],
        metadata={"source": "test"},
    )


def _sample_execution_records() -> list[BacktestExecutionRecord]:
    return [
        BacktestExecutionRecord(
            run_id="test_run",
            timestamp_ms=1,
            topic="execution.order_submitted",
            order_id="order_1",
            signal_id="signal_1",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            status=SimulatedOrderStatus.SUBMITTED,
            payload={"order_id": "order_1"},
            metadata={"source": "test"},
        ),
        BacktestExecutionRecord(
            run_id="test_run",
            timestamp_ms=2,
            topic="execution.order_filled",
            order_id="order_1",
            fill_id="fill_1",
            signal_id="signal_1",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            status=SimulatedOrderStatus.FILLED,
            payload={"order_id": "order_1", "fill_id": "fill_1"},
            metadata={"source": "test"},
        ),
        BacktestExecutionRecord(
            run_id="test_run",
            timestamp_ms=3,
            topic="execution.order_rejected",
            order_id="order_2",
            signal_id="signal_2",
            strategy_name="fake_strategy",
            symbol="BTCUSDT",
            status=SimulatedOrderStatus.REJECTED,
            payload={"order_id": "order_2"},
            metadata={"source": "test"},
        ),
    ]


def _metrics_input(
    *,
    trades: list[SimulatedTrade],
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> MetricsInput:
    return build_metrics_input_from_components(
        initial_balance=10_000.0,
        final_balance=10_300.0,
        final_equity=10_300.0,
        trades=trades,
        positions=[_sample_position()],
        equity_curve=sample_equity_curve,
        signals=sample_signal_records,
        risk_decisions=sample_risk_decisions,
        orders=_sample_orders(),
        fills=[sample_long_fill],
        execution_records=_sample_execution_records(),
        metadata={"run_id": "test_run", "source": "test"},
    )


def _build_result(
    *,
    performance_metrics_config: PerformanceMetricsConfig,
    model_analytics_config: ModelAnalyticsConfig,
    backtest_period: BacktestPeriod,
    trades: list[SimulatedTrade],
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> BacktestResult:
    metrics_input = _metrics_input(
        trades=trades,
        sample_equity_curve=sample_equity_curve,
        sample_signal_records=sample_signal_records,
        sample_risk_decisions=sample_risk_decisions,
        sample_long_fill=sample_long_fill,
    )

    portfolio = PerformanceMetrics(performance_metrics_config).calculate_portfolio_result(metrics_input)
    analytics = BacktestModelAnalyticsEngine(model_analytics_config).analyze(
        ModelAnalyticsInput(
            signals=sample_signal_records,
            risk_decisions=sample_risk_decisions,
            orders=_sample_orders(),
            fills=[sample_long_fill],
            positions=[_sample_position()],
            trades=trades,
            execution_records=_sample_execution_records(),
            metadata={"run_id": "test_run"},
        )
    )

    return BacktestResult(
        run_id="test_run",
        run_name="metrics_analytics_report_test",
        mode=BacktestMode.SINGLE_STRATEGY,
        status=BacktestStatus.COMPLETED,
        period=backtest_period,
        initial_balance=10_000.0,
        final_balance=10_300.0,
        final_equity=10_300.0,
        portfolio=portfolio,
        analytics=analytics,
        signals=list(sample_signal_records),
        risk_decisions=list(sample_risk_decisions),
        execution_records=_sample_execution_records(),
        orders=_sample_orders(),
        fills=[sample_long_fill],
        positions=[_sample_position()],
        trades=list(trades),
        equity_curve=list(sample_equity_curve),
        metadata={"source": "test"},
    )


def _artifact_paths(report) -> dict[BacktestArtifactType, list[Path]]:
    paths: dict[BacktestArtifactType, list[Path]] = {}
    for artifact in report.artifacts:
        paths.setdefault(artifact.artifact_type, []).append(Path(artifact.path))
    return paths


# =============================================================================
# Performance metrics
# =============================================================================


def test_trade_stats_calculator_counts_closed_wins_and_losses(
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)

    stats = TradeStatsCalculator().calculate(trades)

    assert stats.total_trades == 2
    assert stats.closed_trades == 2
    assert stats.winning_trades == 1
    assert stats.losing_trades == 1
    assert stats.win_rate == pytest.approx(50.0)
    assert stats.gross_profit == pytest.approx(9.9)
    assert stats.gross_loss == pytest.approx(5.1)
    assert stats.net_profit == pytest.approx(4.8)
    assert stats.profit_factor == pytest.approx(9.9 / 5.1)
    assert stats.expectancy_r == pytest.approx(0.5)


def test_performance_metrics_calculates_portfolio_result(
    performance_metrics_config: PerformanceMetricsConfig,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)
    data = _metrics_input(
        trades=trades,
        sample_equity_curve=sample_equity_curve,
        sample_signal_records=sample_signal_records,
        sample_risk_decisions=sample_risk_decisions,
        sample_long_fill=sample_long_fill,
    )

    result = PerformanceMetrics(performance_metrics_config).calculate_portfolio_result(data)

    assert result.summary.aggregation == MetricAggregation.SYSTEM
    assert result.summary.key == "system"
    assert result.summary.initial_balance == pytest.approx(10_000.0)
    assert result.summary.final_equity == pytest.approx(10_300.0)
    assert result.summary.net_profit == pytest.approx(300.0)
    assert result.summary.net_profit_pct == pytest.approx(3.0)
    assert result.trade_stats.closed_trades == 2
    assert result.trade_stats.win_rate == pytest.approx(50.0)
    assert result.drawdowns
    assert result.costs.commission >= 0.0
    assert result.equity_curve == sample_equity_curve


def test_performance_metrics_builds_strategy_and_symbol_breakdowns(
    performance_metrics_config: PerformanceMetricsConfig,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)
    data = _metrics_input(
        trades=trades,
        sample_equity_curve=sample_equity_curve,
        sample_signal_records=sample_signal_records,
        sample_risk_decisions=sample_risk_decisions,
        sample_long_fill=sample_long_fill,
    )

    result = PerformanceMetrics(performance_metrics_config).calculate_portfolio_result(data)

    assert "fake_strategy" in result.strategy_results
    assert "BTCUSDT" in result.symbol_results

    strategy_result = result.strategy_results["fake_strategy"]
    symbol_result = result.symbol_results["BTCUSDT"]

    assert strategy_result.summary.aggregation == MetricAggregation.STRATEGY
    assert strategy_result.trade_stats.closed_trades == 2
    assert strategy_result.risk_stats.signals_received == 2
    assert symbol_result.summary.aggregation == MetricAggregation.SYMBOL
    assert symbol_result.trade_stats.closed_trades == 2


def test_risk_execution_and_cost_stats_are_aggregated(
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    risk_stats = RiskStatsCalculator().calculate(
        signals=sample_signal_records,
        risk_decisions=sample_risk_decisions,
    )
    execution_stats = ExecutionStatsCalculator().calculate(
        orders=_sample_orders(),
        fills=[sample_long_fill],
        execution_records=_sample_execution_records(),
    )
    costs = CostStatsCalculator().calculate(
        trades=[],
        positions=[_sample_position()],
        fills=[sample_long_fill],
    )

    assert risk_stats.signals_received == 2
    assert risk_stats.signals_confirmed == 1
    assert risk_stats.signals_blocked == 1
    assert risk_stats.confirmation_rate == pytest.approx(50.0)

    assert execution_stats.orders_submitted >= 1
    assert execution_stats.orders_filled >= 1
    assert execution_stats.orders_rejected >= 1
    assert execution_stats.fills == 1
    assert execution_stats.total_fees == pytest.approx(sample_long_fill.fee)

    assert costs.commission == pytest.approx(sample_long_fill.fee)
    assert costs.slippage == pytest.approx(sample_long_fill.slippage)


def test_drawdown_and_standalone_metric_helpers(
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_equity_curve: list[SimulatedEquityPoint],
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)

    drawdowns, max_drawdown, max_drawdown_pct, average_drawdown = DrawdownCalculator().calculate_drawdowns(
        sample_equity_curve,
        max_periods=10,
    )

    assert drawdowns
    assert max_drawdown > 0.0
    assert max_drawdown_pct > 0.0
    assert average_drawdown >= 0.0

    assert calculate_profit_factor(trades) == pytest.approx(9.9 / 5.1)
    assert calculate_win_rate(trades) == pytest.approx(50.0)
    assert calculate_expectancy(trades) == pytest.approx((9.9 - 5.1) / 2.0)
    assert calculate_max_drawdown_pct(sample_equity_curve) == pytest.approx(max_drawdown_pct)


def test_metrics_input_validation_rejects_invalid_balance() -> None:
    with pytest.raises(MetricInputError):
        MetricsInput(initial_balance=0.0).validate()

    with pytest.raises(MetricInputError):
        MetricsInput(initial_balance=10_000.0, final_equity=-1.0).validate()


# =============================================================================
# Model analytics
# =============================================================================


def test_model_analytics_calculates_signal_quality_and_attribution(
    model_analytics_config: ModelAnalyticsConfig,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)

    analytics = BacktestModelAnalyticsEngine(model_analytics_config).analyze(
        ModelAnalyticsInput(
            signals=sample_signal_records,
            risk_decisions=sample_risk_decisions,
            orders=_sample_orders(),
            fills=[sample_long_fill],
            positions=[_sample_position()],
            trades=trades,
            execution_records=_sample_execution_records(),
        )
    )

    assert analytics.signal_quality.signals_generated == 2
    assert analytics.signal_quality.signals_confirmed == 1
    assert analytics.signal_quality.signals_blocked_by_risk == 1
    assert analytics.signal_quality.signals_executed >= 1
    assert analytics.signal_quality.profitable_signal_rate == pytest.approx(50.0)
    assert analytics.signal_quality.average_signal_pnl == pytest.approx((9.9 - 5.1) / 2.0)

    assert analytics.strategy_attribution
    attribution = analytics.strategy_attribution[0]
    assert attribution.strategy_name == "fake_strategy"
    assert attribution.signals == 2
    assert attribution.blocked_signals == 1
    assert attribution.net_profit == pytest.approx(4.8)


def test_model_analytics_builds_regime_feature_risk_and_execution_diagnostics(
    model_analytics_config: ModelAnalyticsConfig,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)

    analytics = BacktestModelAnalyticsEngine(model_analytics_config).analyze_from_components(
        signals=sample_signal_records,
        risk_decisions=sample_risk_decisions,
        orders=_sample_orders(),
        fills=[sample_long_fill],
        positions=[_sample_position()],
        trades=trades,
        execution_records=_sample_execution_records(),
    )

    regimes = {item.regime: item for item in analytics.regime_performance}
    assert "trend" in regimes
    assert "range" in regimes
    assert regimes["trend"].net_profit == pytest.approx(9.9)
    assert regimes["range"].net_profit == pytest.approx(-5.1)

    assert analytics.feature_stats["total_unique_features"] > 0
    assert "features" in analytics.feature_stats
    assert analytics.risk_decision_stats["total_decisions"] == 2
    assert analytics.risk_decision_stats["approved"] == 1
    assert analytics.risk_decision_stats["blocked"] == 1
    assert analytics.execution_quality_stats["orders"] == 2
    assert analytics.execution_quality_stats["fills"] == 1
    assert analytics.execution_quality_stats["rejected_orders"] == 1


def test_model_analytics_helpers_rank_filter_and_summarize(
    model_analytics_config: ModelAnalyticsConfig,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)
    analytics = BacktestModelAnalyticsEngine(model_analytics_config).analyze_from_components(
        signals=sample_signal_records,
        risk_decisions=sample_risk_decisions,
        orders=_sample_orders(),
        fills=[sample_long_fill],
        trades=trades,
        execution_records=_sample_execution_records(),
    )

    by_profit = rank_strategies_by_profit(analytics.strategy_attribution)
    by_profit_factor = rank_strategies_by_profit_factor(analytics.strategy_attribution)
    weak = find_underperforming_strategies(
        analytics.strategy_attribution,
        min_trades=1,
        min_profit_factor=10.0,
        min_expectancy=10.0,
    )
    summary = summarize_model_analytics(analytics)

    assert by_profit[0].strategy_name == "fake_strategy"
    assert by_profit_factor[0].strategy_name == "fake_strategy"
    assert weak and weak[0].strategy_name == "fake_strategy"
    assert summary["signals_generated"] == 2
    assert summary["signals_confirmed"] == 1
    assert summary["best_strategy"] == "fake_strategy"
    assert summary["features_analyzed"] > 0


# =============================================================================
# Report builder
# =============================================================================


def test_report_builder_renders_markdown_sections(
    performance_metrics_config: PerformanceMetricsConfig,
    model_analytics_config: ModelAnalyticsConfig,
    report_builder_config: ReportBuilderConfig,
    backtest_period: BacktestPeriod,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)
    result = _build_result(
        performance_metrics_config=performance_metrics_config,
        model_analytics_config=model_analytics_config,
        backtest_period=backtest_period,
        trades=trades,
        sample_equity_curve=sample_equity_curve,
        sample_signal_records=sample_signal_records,
        sample_risk_decisions=sample_risk_decisions,
        sample_long_fill=sample_long_fill,
    )

    markdown = ReportBuilder(report_builder_config).render_markdown(result)
    helper_markdown = build_markdown_report(result, config=report_builder_config)

    assert "# Backtest Test Report" in markdown
    assert "## Summary" in markdown
    assert "## Trades" in markdown
    assert "## Risk" in markdown
    assert "## Execution" in markdown
    assert "## Costs" in markdown
    assert "fake_strategy" in markdown
    assert "BTCUSDT" in markdown
    assert helper_markdown == markdown


def test_report_builder_exports_markdown_json_csv_and_events(
    performance_metrics_config: PerformanceMetricsConfig,
    model_analytics_config: ModelAnalyticsConfig,
    report_builder_config: ReportBuilderConfig,
    backtest_period: BacktestPeriod,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    trades = _sample_trades(sample_trade, losing_trade)
    result = _build_result(
        performance_metrics_config=performance_metrics_config,
        model_analytics_config=model_analytics_config,
        backtest_period=backtest_period,
        trades=trades,
        sample_equity_curve=sample_equity_curve,
        sample_signal_records=sample_signal_records,
        sample_risk_decisions=sample_risk_decisions,
        sample_long_fill=sample_long_fill,
    )

    report = ReportBuilder(report_builder_config).build(result)
    paths = _artifact_paths(report)

    assert report.run_id == "test_run"
    assert report.summary is result.portfolio.summary
    assert report.metadata["artifacts"] == len(report.artifacts)
    assert result.reports[-1] is report
    assert result.artifacts

    required_artifacts = {
        BacktestArtifactType.REPORT_MARKDOWN,
        BacktestArtifactType.RESULT_JSON,
        BacktestArtifactType.TRADES_CSV,
        BacktestArtifactType.POSITIONS_CSV,
        BacktestArtifactType.EQUITY_CURVE_CSV,
        BacktestArtifactType.EVENTS_JSONL,
    }

    for artifact_type in required_artifacts:
        assert artifact_type in paths
        assert any(path.exists() for path in paths[artifact_type])

    json_path = paths[BacktestArtifactType.RESULT_JSON][0]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run"
    assert payload["status"] == BacktestStatus.COMPLETED.value

    markdown_path = paths[BacktestArtifactType.REPORT_MARKDOWN][0]
    assert "Backtest Test Report" in markdown_path.read_text(encoding="utf-8")

    events_path = paths[BacktestArtifactType.EVENTS_JSONL][0]
    assert events_path.read_text(encoding="utf-8").strip()


def test_export_backtest_report_helper_returns_saved_report(
    tmp_path: Path,
    performance_metrics_config: PerformanceMetricsConfig,
    model_analytics_config: ModelAnalyticsConfig,
    backtest_period: BacktestPeriod,
    sample_trade: SimulatedTrade,
    losing_trade: SimulatedTrade,
    sample_equity_curve: list[SimulatedEquityPoint],
    sample_signal_records: list[BacktestSignalRecord],
    sample_risk_decisions: list[BacktestRiskDecisionRecord],
    sample_long_fill: SimulatedFill,
) -> None:
    config = ReportBuilderConfig(
        enabled=True,
        output_dir=tmp_path / "helper_reports",
        report_title="Helper Report",
        formats=[ReportFormat.MARKDOWN, ReportFormat.JSON],
        sections={ReportSection.SUMMARY, ReportSection.TRADES, ReportSection.WARNINGS},
        save_result_json=True,
        save_trades_csv=False,
        save_positions_csv=False,
        save_equity_curve_csv=False,
        save_events_jsonl=False,
    )
    result = _build_result(
        performance_metrics_config=performance_metrics_config,
        model_analytics_config=model_analytics_config,
        backtest_period=backtest_period,
        trades=_sample_trades(sample_trade, losing_trade),
        sample_equity_curve=sample_equity_curve,
        sample_signal_records=sample_signal_records,
        sample_risk_decisions=sample_risk_decisions,
        sample_long_fill=sample_long_fill,
    )

    report = export_backtest_report(result, config=config)

    assert report.title == "Helper Report"
    assert len(report.artifacts) == 2
    assert all(Path(artifact.path).exists() for artifact in report.artifacts)


def test_disabled_report_builder_returns_descriptor_without_artifacts(
    tmp_path: Path,
) -> None:
    config = ReportBuilderConfig(
        enabled=False,
        output_dir=tmp_path / "disabled_reports",
        report_title="Disabled Report",
        formats=[ReportFormat.MARKDOWN],
        sections={ReportSection.SUMMARY},
    )
    result = BacktestResult(
        run_id="disabled_run",
        run_name="disabled_report_test",
        status=BacktestStatus.COMPLETED,
        portfolio=type("Portfolio", (), {"summary": PerformanceSummary(net_profit=1.0)})(),
    )

    report = ReportBuilder(config).build(result)

    assert report.run_id == "disabled_run"
    assert report.title == "Disabled Report"
    assert report.artifacts == []
    assert report.metadata["enabled"] is False