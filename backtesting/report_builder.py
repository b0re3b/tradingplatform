"""
Backtest report builder.

This module builds user-facing backtest reports and exports artifacts:
- Markdown report;
- JSON result snapshot;
- trades CSV;
- positions CSV;
- equity curve CSV;
- optional events JSONL.

Important:
- No EventBus usage here.
- No strategy/risk/execution decisions here.
- No live exchange calls here.
- This module formats already calculated backtest results.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from backtesting.config import ReportBuilderConfig
from backtesting.enums import (
    BacktestArtifactType,
    ReportFormat,
    ReportSection,
)
from backtesting.exceptions import (
    ReportArtifactError,
    ReportBuildError,
    ReportFormatError,
)
from backtesting.model_analytics import summarize_model_analytics
from backtesting.models import (
    BacktestArtifact,
    BacktestEvent,
    BacktestExecutionRecord,
    BacktestPositionRecord,
    BacktestReport,
    BacktestResult,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    SimulatedEquityPoint,
    SimulatedPosition,
    SimulatedTrade,
    utcnow,
)


class ReportBuilder:
    """
    Builds reports and exports artifacts for a completed backtest.
    """

    def __init__(
        self,
        config: ReportBuilderConfig | None = None,
    ) -> None:
        self.config = config or ReportBuilderConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        result: BacktestResult,
    ) -> BacktestReport:
        """
        Build configured report formats and export artifacts.
        """

        if not self.config.enabled:
            return BacktestReport(
                run_id=result.run_id,
                title=self.config.report_title,
                format=ReportFormat.MARKDOWN,
                summary=result.portfolio.summary,
                metadata={"enabled": False},
            )

        output_dir = self._run_output_dir(result)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[BacktestArtifact] = []

        for report_format in self.config.formats:
            if report_format == ReportFormat.MARKDOWN:
                artifacts.append(self.build_markdown_report(result, output_dir=output_dir))
                continue

            if report_format == ReportFormat.JSON:
                artifacts.append(self.export_result_json(result, output_dir=output_dir))
                continue

            if report_format == ReportFormat.HTML:
                artifacts.append(self.build_html_report(result, output_dir=output_dir))
                continue

            if report_format == ReportFormat.CSV:
                artifacts.extend(self.export_csv_artifacts(result, output_dir=output_dir))
                continue

            if report_format == ReportFormat.PARQUET:
                artifacts.extend(self.export_parquet_artifacts(result, output_dir=output_dir))
                continue

            raise ReportFormatError(
                "Unsupported report format.",
                details={"format": report_format.value},
            )

        if self.config.save_result_json and not any(
            item.artifact_type == BacktestArtifactType.RESULT_JSON
            for item in artifacts
        ):
            artifacts.append(self.export_result_json(result, output_dir=output_dir))

        if self.config.save_trades_csv:
            artifacts.append(self.export_trades_csv(result.trades, output_dir / "trades.csv"))

        if self.config.save_positions_csv:
            artifacts.append(self.export_positions_csv(result.positions, output_dir / "positions.csv"))

        if self.config.save_equity_curve_csv:
            artifacts.append(self.export_equity_curve_csv(result.equity_curve, output_dir / "equity_curve.csv"))

        if self.config.save_events_jsonl:
            artifacts.append(
                self.export_events_jsonl(
                    events=[
                        *result.execution_records,
                        *result.position_records,
                        *result.risk_decisions,
                        *result.signals,
                    ],
                    path=output_dir / "events.jsonl",
                )
            )

        report = BacktestReport(
            run_id=result.run_id,
            title=self.config.report_title,
            format=self.config.formats[0] if self.config.formats else ReportFormat.MARKDOWN,
            path=str(artifacts[0].path) if artifacts else None,
            summary=result.portfolio.summary,
            artifacts=artifacts,
            created_at=utcnow(),
            metadata={
                "run_name": result.run_name,
                "status": result.status.value,
                "artifacts": len(artifacts),
            },
        )

        result.reports.append(report)
        result.artifacts.extend(artifacts)

        return report

    def build_markdown_report(
        self,
        result: BacktestResult,
        *,
        output_dir: str | Path | None = None,
    ) -> BacktestArtifact:
        """
        Build Markdown report artifact.
        """

        output_path = Path(output_dir or self._run_output_dir(result)) / "report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            content = self.render_markdown(result)
            output_path.write_text(content, encoding="utf-8")

            return BacktestArtifact(
                artifact_type=BacktestArtifactType.REPORT_MARKDOWN,
                path=str(output_path),
                format=ReportFormat.MARKDOWN,
                size_bytes=output_path.stat().st_size,
                metadata={"run_id": result.run_id},
            )

        except Exception as exc:
            raise ReportBuildError(
                "Failed to build Markdown report.",
                details={
                    "path": str(output_path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    def build_html_report(
        self,
        result: BacktestResult,
        *,
        output_dir: str | Path | None = None,
    ) -> BacktestArtifact:
        """
        Build simple self-contained HTML report artifact.
        """

        output_path = Path(output_dir or self._run_output_dir(result)) / "report.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            markdown = self.render_markdown(result)
            html = self._markdown_to_simple_html(markdown)
            output_path.write_text(html, encoding="utf-8")

            return BacktestArtifact(
                artifact_type=BacktestArtifactType.REPORT_HTML,
                path=str(output_path),
                format=ReportFormat.HTML,
                size_bytes=output_path.stat().st_size,
                metadata={"run_id": result.run_id},
            )

        except Exception as exc:
            raise ReportBuildError(
                "Failed to build HTML report.",
                details={
                    "path": str(output_path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    def export_result_json(
        self,
        result: BacktestResult,
        *,
        output_dir: str | Path | None = None,
    ) -> BacktestArtifact:
        """
        Export full BacktestResult snapshot as JSON.
        """

        output_path = Path(output_dir or self._run_output_dir(result)) / "result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = result.to_dict()
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            return BacktestArtifact(
                artifact_type=BacktestArtifactType.RESULT_JSON,
                path=str(output_path),
                format=ReportFormat.JSON,
                size_bytes=output_path.stat().st_size,
                metadata={"run_id": result.run_id},
            )

        except Exception as exc:
            raise ReportArtifactError(
                "Failed to export result JSON.",
                details={
                    "path": str(output_path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    def export_csv_artifacts(
        self,
        result: BacktestResult,
        *,
        output_dir: str | Path | None = None,
    ) -> list[BacktestArtifact]:
        """
        Export CSV artifacts.
        """

        base = Path(output_dir or self._run_output_dir(result))
        base.mkdir(parents=True, exist_ok=True)

        artifacts: list[BacktestArtifact] = []

        artifacts.append(self.export_trades_csv(result.trades, base / "trades.csv"))
        artifacts.append(self.export_positions_csv(result.positions, base / "positions.csv"))
        artifacts.append(self.export_equity_curve_csv(result.equity_curve, base / "equity_curve.csv"))

        return artifacts

    def export_parquet_artifacts(
        self,
        result: BacktestResult,
        *,
        output_dir: str | Path | None = None,
    ) -> list[BacktestArtifact]:
        """
        Export parquet artifacts.

        Requires pandas + parquet engine.
        """

        base = Path(output_dir or self._run_output_dir(result))
        base.mkdir(parents=True, exist_ok=True)

        try:
            import pandas as pd
        except Exception as exc:
            raise ReportFormatError(
                "Parquet export requires pandas and a parquet engine.",
                details={"output_dir": str(base)},
            ) from exc

        artifacts: list[BacktestArtifact] = []

        exports = [
            (
                BacktestArtifactType.TRADES_CSV,
                base / "trades.parquet",
                [self._to_flat_dict(item) for item in result.trades],
            ),
            (
                BacktestArtifactType.POSITIONS_CSV,
                base / "positions.parquet",
                [self._to_flat_dict(item) for item in result.positions],
            ),
            (
                BacktestArtifactType.EQUITY_CURVE_CSV,
                base / "equity_curve.parquet",
                [self._to_flat_dict(item) for item in result.equity_curve],
            ),
        ]

        for artifact_type, path, rows in exports:
            try:
                pd.DataFrame(rows).to_parquet(path, index=False)
                artifacts.append(
                    BacktestArtifact(
                        artifact_type=artifact_type,
                        path=str(path),
                        format=ReportFormat.PARQUET,
                        size_bytes=path.stat().st_size,
                        metadata={"run_id": result.run_id},
                    )
                )
            except Exception as exc:
                raise ReportArtifactError(
                    "Failed to export parquet artifact.",
                    details={
                        "path": str(path),
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                ) from exc

        return artifacts

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def render_markdown(self, result: BacktestResult) -> str:
        """
        Render complete Markdown report.
        """

        sections: list[str] = []

        sections.append(self._render_header(result))

        if ReportSection.SUMMARY in self.config.sections:
            sections.append(self._render_summary(result))

        if ReportSection.EQUITY_CURVE in self.config.sections:
            sections.append(self._render_equity_curve_summary(result))

        if ReportSection.DRAWDOWN in self.config.sections:
            sections.append(self._render_drawdown(result))

        if ReportSection.TRADES in self.config.sections:
            sections.append(self._render_trades(result))

        if ReportSection.POSITIONS in self.config.sections:
            sections.append(self._render_positions(result))

        if ReportSection.STRATEGIES in self.config.sections:
            sections.append(self._render_strategies(result))

        if ReportSection.SYMBOLS in self.config.sections:
            sections.append(self._render_symbols(result))

        if ReportSection.RISK in self.config.sections:
            sections.append(self._render_risk(result))

        if ReportSection.EXECUTION in self.config.sections:
            sections.append(self._render_execution(result))

        if ReportSection.COSTS in self.config.sections:
            sections.append(self._render_costs(result))

        if ReportSection.SIGNALS in self.config.sections:
            sections.append(self._render_signals(result))

        if ReportSection.REGIMES in self.config.sections:
            sections.append(self._render_regimes(result))

        if ReportSection.WARNINGS in self.config.sections:
            sections.append(self._render_warnings(result))

        if self.config.include_config_snapshot:
            sections.append(self._render_config_snapshot(result))

        if self.config.include_metadata:
            sections.append(self._render_metadata(result))

        return "\n\n".join(section for section in sections if section.strip()) + "\n"

    def _render_header(self, result: BacktestResult) -> str:
        period = result.period

        if period is not None:
            period_text = f"{period.start.isoformat()} → {period.end.isoformat()}"
        else:
            period_text = "n/a"

        return "\n".join(
            [
                f"# {self.config.report_title}",
                "",
                f"- **Run ID:** `{result.run_id}`",
                f"- **Run name:** `{result.run_name}`",
                f"- **Mode:** `{result.mode.value}`",
                f"- **Status:** `{result.status.value}`",
                f"- **Period:** {period_text}",
                f"- **Started at:** {result.started_at.isoformat() if result.started_at else 'n/a'}",
                f"- **Finished at:** {result.finished_at.isoformat() if result.finished_at else 'n/a'}",
                f"- **Duration:** {self._fmt_number(result.duration_seconds)} sec",
            ]
        )

    def _render_summary(self, result: BacktestResult) -> str:
        summary = result.portfolio.summary

        rows = [
            ("Initial balance", self._fmt_money(result.initial_balance)),
            ("Final balance", self._fmt_money(result.final_balance)),
            ("Final equity", self._fmt_money(result.final_equity)),
            ("Net profit", self._fmt_money(result.net_profit)),
            ("Net profit %", self._fmt_pct(result.net_profit_pct)),
            ("Total trades", str(result.total_trades)),
            ("Win rate", self._fmt_pct(summary.win_rate)),
            ("Profit factor", self._fmt_number(summary.profit_factor)),
            ("Expectancy", self._fmt_money(summary.expectancy)),
            ("Max drawdown", self._fmt_money(summary.max_drawdown)),
            ("Max drawdown %", self._fmt_pct(summary.max_drawdown_pct)),
            ("Sharpe", self._fmt_optional(summary.sharpe_ratio)),
            ("Sortino", self._fmt_optional(summary.sortino_ratio)),
            ("Calmar", self._fmt_optional(summary.calmar_ratio)),
            ("Recovery factor", self._fmt_optional(summary.recovery_factor)),
            ("Exposure time", self._fmt_pct(summary.exposure_time_pct)),
        ]

        return "## Summary\n\n" + self._render_key_value_table(rows)

    def _render_equity_curve_summary(self, result: BacktestResult) -> str:
        points = result.equity_curve

        if not points:
            return "## Equity Curve\n\nNo equity curve data."

        first = points[0]
        last = points[-1]
        high = max(points, key=lambda item: item.equity)
        low = min(points, key=lambda item: item.equity)

        rows = [
            ("Equity points", str(len(points))),
            ("Start equity", self._fmt_money(first.equity)),
            ("End equity", self._fmt_money(last.equity)),
            ("Highest equity", self._fmt_money(high.equity)),
            ("Lowest equity", self._fmt_money(low.equity)),
            ("Last timestamp", str(last.timestamp_ms)),
            ("Open positions at end", str(last.open_positions)),
        ]

        return "## Equity Curve\n\n" + self._render_key_value_table(rows)

    def _render_drawdown(self, result: BacktestResult) -> str:
        drawdowns = result.portfolio.drawdowns

        if not drawdowns:
            return "## Drawdown\n\nNo drawdown periods."

        lines = [
            "## Drawdown",
            "",
            "| # | Drawdown | Drawdown % | Peak equity | Valley equity | Recovered | Duration sec |",
            "|---:|---:|---:|---:|---:|:---:|---:|",
        ]

        for index, item in enumerate(drawdowns[:20], start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{self._fmt_money(item.drawdown)} | "
                f"{self._fmt_pct(item.drawdown_pct)} | "
                f"{self._fmt_money(item.peak_equity)} | "
                f"{self._fmt_money(item.valley_equity)} | "
                f"{'yes' if item.is_recovered else 'no'} | "
                f"{self._fmt_number(item.duration_seconds)} |"
            )

        return "\n".join(lines)

    def _render_trades(self, result: BacktestResult) -> str:
        trades = result.trades

        if not trades:
            return "## Trades\n\nNo trades."

        stats = result.portfolio.trade_stats
        rows = [
            ("Total trades", str(stats.total_trades)),
            ("Closed trades", str(stats.closed_trades)),
            ("Open trades", str(stats.open_trades)),
            ("Winning trades", str(stats.winning_trades)),
            ("Losing trades", str(stats.losing_trades)),
            ("Breakeven trades", str(stats.breakeven_trades)),
            ("Average trade", self._fmt_money(stats.average_trade)),
            ("Average win", self._fmt_money(stats.average_win)),
            ("Average loss", self._fmt_money(stats.average_loss)),
            ("Best trade", self._fmt_money(stats.best_trade)),
            ("Worst trade", self._fmt_money(stats.worst_trade)),
            ("Average holding time", f"{self._fmt_number(stats.average_holding_time_seconds)} sec"),
        ]

        lines = [
            "## Trades",
            "",
            self._render_key_value_table(rows),
            "",
        ]

        if self.config.include_full_trade_list:
            lines.extend(self._render_trade_table(trades[: self.config.max_trades_in_markdown]))

            if len(trades) > self.config.max_trades_in_markdown:
                lines.append("")
                lines.append(
                    f"_Showing first {self.config.max_trades_in_markdown} of {len(trades)} trades._"
                )

        return "\n".join(lines)

    def _render_trade_table(self, trades: list[SimulatedTrade]) -> list[str]:
        lines = [
            "| # | Strategy | Symbol | Side | Entry | Exit | Net PnL | R | Outcome | Reason |",
            "|---:|---|---|---|---:|---:|---:|---:|---|---|",
        ]

        for index, trade in enumerate(trades, start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{self._safe_cell(trade.strategy_name)} | "
                f"{self._safe_cell(trade.symbol)} | "
                f"{self._safe_cell(trade.side)} | "
                f"{self._fmt_number(trade.entry_price)} | "
                f"{self._fmt_number(trade.exit_price)} | "
                f"{self._fmt_money(trade.net_pnl)} | "
                f"{self._fmt_optional(trade.r_multiple)} | "
                f"{trade.outcome.value} | "
                f"{self._safe_cell(trade.close_reason)} |"
            )

        return lines

    def _render_positions(self, result: BacktestResult) -> str:
        positions = result.positions

        if not positions:
            return "## Positions\n\nNo positions."

        open_positions = [item for item in positions if item.is_open]
        closed_positions = [item for item in positions if not item.is_open]

        rows = [
            ("Total positions", str(len(positions))),
            ("Open positions", str(len(open_positions))),
            ("Closed positions", str(len(closed_positions))),
            ("Liquidated positions", str(len([item for item in positions if item.status.value == 'liquidated']))),
        ]

        lines = [
            "## Positions",
            "",
            self._render_key_value_table(rows),
            "",
            "| # | Strategy | Symbol | Side | Status | Qty | Entry | Mark/Exit | Net realized PnL |",
            "|---:|---|---|---|---|---:|---:|---:|---:|",
        ]

        for index, position in enumerate(positions[:100], start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{self._safe_cell(position.strategy_name)} | "
                f"{self._safe_cell(position.symbol)} | "
                f"{self._safe_cell(position.side)} | "
                f"{position.status.value} | "
                f"{self._fmt_number(position.quantity)} | "
                f"{self._fmt_number(position.entry_price)} | "
                f"{self._fmt_number(position.exit_price or position.mark_price)} | "
                f"{self._fmt_money(position.net_realized_pnl)} |"
            )

        if len(positions) > 100:
            lines.append("")
            lines.append(f"_Showing first 100 of {len(positions)} positions._")

        return "\n".join(lines)

    def _render_strategies(self, result: BacktestResult) -> str:
        strategy_results = result.portfolio.strategy_results
        attributions = result.analytics.strategy_attribution

        if not strategy_results and not attributions:
            return "## Strategy Breakdown\n\nNo strategy breakdown."

        lines = [
            "## Strategy Breakdown",
            "",
            "| Strategy | Net profit | Profit share | Trades | Win rate | Profit factor | Expectancy | Signals | Blocked |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        if attributions:
            for item in attributions:
                lines.append(
                    "| "
                    f"{self._safe_cell(item.strategy_name)} | "
                    f"{self._fmt_money(item.net_profit)} | "
                    f"{self._fmt_pct(item.profit_share_pct)} | "
                    f"{item.trades} | "
                    f"{self._fmt_pct(item.win_rate)} | "
                    f"{self._fmt_number(item.profit_factor)} | "
                    f"{self._fmt_money(item.expectancy)} | "
                    f"{item.signals} | "
                    f"{item.blocked_signals} |"
                )
        else:
            for strategy_name, strategy_result in strategy_results.items():
                summary = strategy_result.summary
                lines.append(
                    "| "
                    f"{self._safe_cell(strategy_name)} | "
                    f"{self._fmt_money(summary.net_profit)} | "
                    f"n/a | "
                    f"{summary.total_trades} | "
                    f"{self._fmt_pct(summary.win_rate)} | "
                    f"{self._fmt_number(summary.profit_factor)} | "
                    f"{self._fmt_money(summary.expectancy)} | "
                    f"{len(strategy_result.signals)} | "
                    f"{strategy_result.risk_stats.signals_blocked} |"
                )

        return "\n".join(lines)

    def _render_symbols(self, result: BacktestResult) -> str:
        symbol_results = result.portfolio.symbol_results

        if not symbol_results:
            return "## Symbol Breakdown\n\nNo symbol breakdown."

        lines = [
            "## Symbol Breakdown",
            "",
            "| Symbol | Net profit | Trades | Win rate | Profit factor | Max drawdown % |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for symbol, symbol_result in symbol_results.items():
            summary = symbol_result.summary
            lines.append(
                "| "
                f"{self._safe_cell(symbol)} | "
                f"{self._fmt_money(summary.net_profit)} | "
                f"{summary.total_trades} | "
                f"{self._fmt_pct(summary.win_rate)} | "
                f"{self._fmt_number(summary.profit_factor)} | "
                f"{self._fmt_pct(summary.max_drawdown_pct)} |"
            )

        return "\n".join(lines)

    def _render_risk(self, result: BacktestResult) -> str:
        stats = result.portfolio.risk_stats
        analytics = result.analytics.risk_decision_stats or {}

        rows = [
            ("Signals received", str(stats.signals_received)),
            ("Signals confirmed", str(stats.signals_confirmed)),
            ("Signals blocked", str(stats.signals_blocked)),
            ("Confirmation rate", self._fmt_pct(stats.confirmation_rate)),
            ("Block rate", self._fmt_pct(stats.block_rate)),
            ("Position blocked events", str(stats.position_blocked_events)),
            ("Kill switch events", str(stats.kill_switch_events)),
            ("Limit warnings", str(stats.limit_warnings)),
            ("Max margin used", self._fmt_money(stats.max_margin_used)),
            ("Max exposure", self._fmt_money(stats.max_exposure)),
            ("Max leverage used", self._fmt_number(stats.max_leverage_used)),
            ("Reservations created", str(stats.reservations_created)),
            ("Reservations released", str(stats.reservations_released)),
            ("Reservations expired", str(stats.reservations_expired)),
        ]

        lines = [
            "## Risk",
            "",
            self._render_key_value_table(rows),
        ]

        blocked_reasons = analytics.get("blocked_reason_counts") if isinstance(analytics, dict) else None

        if blocked_reasons:
            lines.extend(
                [
                    "",
                    "### Blocked Reasons",
                    "",
                    "| Reason | Count |",
                    "|---|---:|",
                ]
            )

            for reason, count in sorted(blocked_reasons.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"| {self._safe_cell(reason)} | {count} |")

        return "\n".join(lines)

    def _render_execution(self, result: BacktestResult) -> str:
        stats = result.portfolio.execution_stats
        analytics = result.analytics.execution_quality_stats or {}

        rows = [
            ("Orders submitted", str(stats.orders_submitted)),
            ("Orders accepted", str(stats.orders_accepted)),
            ("Orders rejected", str(stats.orders_rejected)),
            ("Orders cancelled", str(stats.orders_cancelled)),
            ("Orders filled", str(stats.orders_filled)),
            ("Orders partially filled", str(stats.orders_partially_filled)),
            ("Fills", str(stats.fills)),
            ("Rejection rate", self._fmt_pct(stats.rejection_rate)),
            ("Fill rate", self._fmt_pct(stats.fill_rate)),
            ("Partial fill rate", self._fmt_pct(stats.partial_fill_rate)),
            ("Average slippage", self._fmt_money(stats.average_slippage)),
            ("Average slippage bps", self._fmt_number(stats.average_slippage_bps)),
            ("Average latency ms", self._fmt_number(stats.average_latency_ms)),
            ("Total fees", self._fmt_money(stats.total_fees)),
            ("Total slippage", self._fmt_money(stats.total_slippage)),
        ]

        lines = [
            "## Execution",
            "",
            self._render_key_value_table(rows),
        ]

        rejected_reasons = analytics.get("rejected_reason_counts") if isinstance(analytics, dict) else None

        if rejected_reasons:
            lines.extend(
                [
                    "",
                    "### Rejected Order Reasons",
                    "",
                    "| Reason | Count |",
                    "|---|---:|",
                ]
            )

            for reason, count in sorted(rejected_reasons.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"| {self._safe_cell(reason)} | {count} |")

        return "\n".join(lines)

    def _render_costs(self, result: BacktestResult) -> str:
        costs = result.portfolio.costs

        rows = [
            ("Commission", self._fmt_money(costs.commission)),
            ("Slippage", self._fmt_money(costs.slippage)),
            ("Spread cost", self._fmt_money(costs.spread_cost)),
            ("Funding paid", self._fmt_money(costs.funding_paid)),
            ("Funding received", self._fmt_money(costs.funding_received)),
            ("Net funding", self._fmt_money(costs.net_funding)),
            ("Borrow cost", self._fmt_money(costs.borrow_cost)),
            ("Liquidation penalty", self._fmt_money(costs.liquidation_penalty)),
            ("Other costs", self._fmt_money(costs.other_costs)),
            ("Total cost", self._fmt_money(costs.total_cost)),
        ]

        return "## Costs\n\n" + self._render_key_value_table(rows)

    def _render_signals(self, result: BacktestResult) -> str:
        signals = result.signals
        quality = result.analytics.signal_quality

        rows = [
            ("Signals generated", str(quality.signals_generated)),
            ("Signals confirmed", str(quality.signals_confirmed)),
            ("Signals blocked by risk", str(quality.signals_blocked_by_risk)),
            ("Signals executed", str(quality.signals_executed)),
            ("Profitable signals", str(quality.signals_profitable)),
            ("Unprofitable signals", str(quality.signals_unprofitable)),
            ("Confirmation rate", self._fmt_pct(quality.confirmation_rate)),
            ("Execution rate", self._fmt_pct(quality.execution_rate)),
            ("Profitable signal rate", self._fmt_pct(quality.profitable_signal_rate)),
            ("Average signal PnL", self._fmt_money(quality.average_signal_pnl)),
            ("Average signal R", self._fmt_optional(quality.average_signal_r)),
        ]

        lines = [
            "## Signals",
            "",
            self._render_key_value_table(rows),
        ]

        if self.config.include_full_signal_list and signals:
            lines.extend(
                [
                    "",
                    "| # | Strategy | Symbol | Side | Confidence | Outcome | PnL |",
                    "|---:|---|---|---|---:|---|---:|",
                ]
            )

            for index, signal in enumerate(signals[: self.config.max_signals_in_markdown], start=1):
                lines.append(
                    "| "
                    f"{index} | "
                    f"{self._safe_cell(signal.strategy_name)} | "
                    f"{self._safe_cell(signal.symbol)} | "
                    f"{self._safe_cell(signal.side)} | "
                    f"{self._fmt_optional(signal.confidence)} | "
                    f"{signal.outcome.value} | "
                    f"{self._fmt_money(signal.pnl)} |"
                )

        return "\n".join(lines)

    def _render_regimes(self, result: BacktestResult) -> str:
        regimes = result.analytics.regime_performance

        if not regimes:
            return "## Regime Performance\n\nNo regime analytics."

        lines = [
            "## Regime Performance",
            "",
            "| Regime | Trades | Net profit | Win rate | Profit factor | Max drawdown % | Avg R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        for regime in regimes:
            lines.append(
                "| "
                f"{self._safe_cell(regime.regime)} | "
                f"{regime.trades} | "
                f"{self._fmt_money(regime.net_profit)} | "
                f"{self._fmt_pct(regime.win_rate)} | "
                f"{self._fmt_number(regime.profit_factor)} | "
                f"{self._fmt_pct(regime.max_drawdown_pct)} | "
                f"{self._fmt_optional(regime.average_r)} |"
            )

        feature_stats = result.analytics.feature_stats

        if isinstance(feature_stats, dict) and feature_stats.get("top_positive"):
            lines.extend(
                [
                    "",
                    "### Top Positive Features",
                    "",
                    "| Feature | Total | Profitable rate | Edge score |",
                    "|---|---:|---:|---:|",
                ]
            )

            for feature, stats in list(feature_stats["top_positive"].items())[:10]:
                lines.append(
                    "| "
                    f"{self._safe_cell(feature)} | "
                    f"{stats.get('total', 0)} | "
                    f"{self._fmt_pct(stats.get('profitable_rate', 0.0))} | "
                    f"{self._fmt_number(stats.get('edge_score', 0.0))} |"
                )

        return "\n".join(lines)

    def _render_warnings(self, result: BacktestResult) -> str:
        warnings = result.warnings[: self.config.max_warnings_in_report]
        analytics_warnings = result.analytics.warnings

        if not warnings and not analytics_warnings:
            return "## Warnings\n\nNo warnings."

        lines = [
            "## Warnings",
            "",
        ]

        for warning in warnings:
            lines.append(
                f"- **{warning.level.value}**"
                f"{f' `{warning.code}`' if warning.code else ''}: "
                f"{warning.message}"
            )

        for warning in analytics_warnings:
            lines.append(f"- **analytics**: {warning}")

        if len(result.warnings) > self.config.max_warnings_in_report:
            lines.append("")
            lines.append(
                f"_Showing first {self.config.max_warnings_in_report} of {len(result.warnings)} warnings._"
            )

        return "\n".join(lines)

    def _render_config_snapshot(self, result: BacktestResult) -> str:
        config = result.metadata.get("config")

        if not config:
            return ""

        return "\n".join(
            [
                "## Config Snapshot",
                "",
                "```json",
                json.dumps(config, ensure_ascii=False, indent=2, default=str),
                "```",
            ]
        )

    def _render_metadata(self, result: BacktestResult) -> str:
        summary = summarize_model_analytics(result.analytics)

        payload = {
            "result_metadata": result.metadata,
            "analytics_summary": summary,
            "dataset_info": result.dataset_info.to_dict() if result.dataset_info else None,
            "simulation_models": result.simulation_models.to_dict(),
        }

        return "\n".join(
            [
                "## Metadata",
                "",
                "```json",
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                "```",
            ]
        )

    # ------------------------------------------------------------------
    # CSV / JSONL exports
    # ------------------------------------------------------------------

    def export_trades_csv(
        self,
        trades: list[SimulatedTrade],
        path: str | Path,
    ) -> BacktestArtifact:
        rows = [self._to_flat_dict(item) for item in trades]
        return self._write_csv_artifact(
            rows=rows,
            path=path,
            artifact_type=BacktestArtifactType.TRADES_CSV,
        )

    def export_positions_csv(
        self,
        positions: list[SimulatedPosition],
        path: str | Path,
    ) -> BacktestArtifact:
        rows = [self._to_flat_dict(item) for item in positions]
        return self._write_csv_artifact(
            rows=rows,
            path=path,
            artifact_type=BacktestArtifactType.POSITIONS_CSV,
        )

    def export_equity_curve_csv(
        self,
        equity_curve: list[SimulatedEquityPoint],
        path: str | Path,
    ) -> BacktestArtifact:
        rows = [self._to_flat_dict(item) for item in equity_curve]
        return self._write_csv_artifact(
            rows=rows,
            path=path,
            artifact_type=BacktestArtifactType.EQUITY_CURVE_CSV,
        )

    def export_events_jsonl(
        self,
        events: list[
            BacktestEvent
            | BacktestExecutionRecord
            | BacktestPositionRecord
            | BacktestRiskDecisionRecord
            | BacktestSignalRecord
        ],
        path: str | Path,
    ) -> BacktestArtifact:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(self._to_json_safe(event), ensure_ascii=False, default=str))
                    handle.write("\n")

            return BacktestArtifact(
                artifact_type=BacktestArtifactType.EVENTS_JSONL,
                path=str(output_path),
                format=ReportFormat.JSON,
                size_bytes=output_path.stat().st_size,
            )

        except Exception as exc:
            raise ReportArtifactError(
                "Failed to export events JSONL.",
                details={
                    "path": str(output_path),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    def _write_csv_artifact(
        self,
        *,
        rows: list[dict[str, Any]],
        path: str | Path,
        artifact_type: BacktestArtifactType,
    ) -> BacktestArtifact:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not rows:
                output_path.write_text("", encoding="utf-8")
            else:
                fieldnames = sorted({key for row in rows for key in row.keys()})

                with output_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            return BacktestArtifact(
                artifact_type=artifact_type,
                path=str(output_path),
                format=ReportFormat.CSV,
                size_bytes=output_path.stat().st_size,
            )

        except Exception as exc:
            raise ReportArtifactError(
                "Failed to write CSV artifact.",
                details={
                    "path": str(output_path),
                    "artifact_type": artifact_type.value,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            ) from exc

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _run_output_dir(self, result: BacktestResult) -> Path:
        safe_run_name = self._safe_filename(result.run_name or "backtest")
        safe_run_id = self._safe_filename(result.run_id)
        return Path(self.config.output_dir) / f"{safe_run_name}_{safe_run_id}"

    @staticmethod
    def _safe_filename(value: str) -> str:
        allowed = []
        for char in value:
            if char.isalnum() or char in {"-", "_", "."}:
                allowed.append(char)
            else:
                allowed.append("_")
        return "".join(allowed).strip("_") or "backtest"

    @staticmethod
    def _render_key_value_table(rows: list[tuple[str, str]]) -> str:
        lines = [
            "| Metric | Value |",
            "|---|---:|",
        ]

        for key, value in rows:
            lines.append(f"| {key} | {value} |")

        return "\n".join(lines)

    @staticmethod
    def _fmt_number(value: Any, digits: int = 4) -> str:
        if value is None:
            return "n/a"

        try:
            number = float(value)
        except Exception:
            return str(value)

        return f"{number:,.{digits}f}"

    @staticmethod
    def _fmt_money(value: Any, digits: int = 2) -> str:
        if value is None:
            return "n/a"

        try:
            number = float(value)
        except Exception:
            return str(value)

        return f"{number:,.{digits}f}"

    @staticmethod
    def _fmt_pct(value: Any, digits: int = 2) -> str:
        if value is None:
            return "n/a"

        try:
            number = float(value)
        except Exception:
            return str(value)

        return f"{number:,.{digits}f}%"

    @classmethod
    def _fmt_optional(cls, value: Any, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        return cls._fmt_number(value, digits=digits)

    @staticmethod
    def _safe_cell(value: Any) -> str:
        if value is None:
            return "n/a"

        text = str(value)
        text = text.replace("|", "\\|")
        text = text.replace("\n", " ")
        return text

    @classmethod
    def _to_flat_dict(cls, value: Any) -> dict[str, Any]:
        payload = cls._to_json_safe(value)

        if not isinstance(payload, dict):
            return {"value": payload}

        result: dict[str, Any] = {}

        def flatten(prefix: str, item: Any) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    nested_key = f"{prefix}.{key}" if prefix else str(key)
                    flatten(nested_key, nested)
                return

            if isinstance(item, list):
                result[prefix] = json.dumps(item, ensure_ascii=False, default=str)
                return

            result[prefix] = item

        flatten("", payload)
        return result

    @classmethod
    def _to_json_safe(cls, value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return value.to_dict()

        if is_dataclass(value):
            return cls._to_json_safe(asdict(value))

        if isinstance(value, dict):
            return {str(key): cls._to_json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [cls._to_json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [cls._to_json_safe(item) for item in value]

        if isinstance(value, set):
            return [cls._to_json_safe(item) for item in sorted(value, key=str)]

        if isinstance(value, datetime):
            return value.isoformat()

        if hasattr(value, "value"):
            return value.value

        return value

    @staticmethod
    def _markdown_to_simple_html(markdown: str) -> str:
        """
        Minimal Markdown-to-HTML fallback.

        This intentionally avoids external dependencies. It preserves the
        Markdown body inside <pre> for readability. A richer renderer can be
        added later if needed.
        """

        escaped = (
            markdown.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        return "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8">',
                "<title>Backtest Report</title>",
                "<style>",
                "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }",
                "pre { white-space: pre-wrap; line-height: 1.45; }",
                "</style>",
                "</head>",
                "<body>",
                "<pre>",
                escaped,
                "</pre>",
                "</body>",
                "</html>",
            ]
        )


# ============================================================================
# Convenience helpers
# ============================================================================


def build_markdown_report(
    result: BacktestResult,
    *,
    config: ReportBuilderConfig | None = None,
) -> str:
    """
    Render Markdown report string without writing files.
    """

    builder = ReportBuilder(config)
    return builder.render_markdown(result)


def export_backtest_report(
    result: BacktestResult,
    *,
    config: ReportBuilderConfig | None = None,
) -> BacktestReport:
    """
    Build report artifacts using ReportBuilder.
    """

    builder = ReportBuilder(config)
    return builder.build(result)


__all__ = [
    "ReportBuilder",
    "build_markdown_report",
    "export_backtest_report",
]