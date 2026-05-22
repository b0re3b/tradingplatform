from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from backtesting.exceptions import BacktestReportError
from backtesting.utils import ensure_dir, safe_float


class BacktestReportBuilder:
    """Build a detailed human-readable HTML/Markdown report with analytics charts."""

    def __init__(self, *, output_dir: Path, report_format: str = "html") -> None:
        self._output_dir = ensure_dir(output_dir)
        self._charts_dir = ensure_dir(output_dir / "charts")
        self._report_format = report_format

    def build(
        self,
        *,
        metrics: dict[str, Any],
        equity_series: list[dict[str, Any]],
        drawdown_series: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        portfolio: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> Path:
        try:
            charts = self._build_charts(
                metrics=metrics,
                equity_series=equity_series,
                drawdown_series=drawdown_series,
                trades=trades,
                orders=orders,
                signals=signals,
                portfolio=portfolio,
            )

            (self._output_dir / "metadata.json").write_text(
                json.dumps(self._json_safe(metadata), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (self._output_dir / "metrics.json").write_text(
                json.dumps(self._json_safe(metrics), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            markdown_path = self._output_dir / "metrics.md"
            markdown_path.write_text(
                self._markdown(metrics=metrics, metadata=metadata, charts=charts),
                encoding="utf-8",
            )

            html_path = self._output_dir / "report.html"
            html_path.write_text(
                self._html(metrics=metrics, metadata=metadata, charts=charts),
                encoding="utf-8",
            )

            return markdown_path if self._report_format == "markdown" else html_path

        except Exception as exc:
            raise BacktestReportError(f"Failed to build backtest report: {exc}") from exc

    def _build_charts(
        self,
        *,
        metrics: dict[str, Any],
        equity_series: list[dict[str, Any]],
        drawdown_series: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        portfolio: list[dict[str, Any]],
    ) -> dict[str, str]:
        charts: dict[str, str] = {}

        charts["equity_curve"] = self._line(
            rows=equity_series,
            y_key="equity",
            title="Portfolio equity curve",
            ylabel="Equity, USD",
            filename="equity_curve.png",
        )
        charts["balance_curve"] = self._line(
            rows=equity_series,
            y_key="balance",
            title="Balance curve",
            ylabel="Balance, USD",
            filename="balance_curve.png",
        )
        charts["drawdown_curve"] = self._line(
            rows=drawdown_series,
            y_key="drawdown",
            title="Drawdown curve",
            ylabel="Drawdown, USD",
            filename="drawdown_curve.png",
        )
        charts["drawdown_pct"] = self._line(
            rows=drawdown_series,
            y_key="drawdown_pct",
            title="Drawdown percent",
            ylabel="Drawdown, %",
            filename="drawdown_pct.png",
        )
        charts["trade_pnl"] = self._trade_pnl(trades)
        charts["cumulative_pnl"] = self._cumulative_pnl(trades)
        charts["pnl_distribution"] = self._histogram(
            values=[safe_float(t.get("realized_pnl") or t.get("net_pnl")) for t in trades],
            title="Trade PnL distribution",
            xlabel="PnL, USD",
            filename="pnl_distribution.png",
        )
        charts["holding_time_distribution"] = self._histogram(
            values=[safe_float(t.get("holding_ms")) / 60000.0 for t in trades],
            title="Holding time distribution",
            xlabel="Holding time, minutes",
            filename="holding_time_distribution.png",
        )
        charts["pnl_by_symbol"] = self._bar(
            data=metrics.get("pnl_by_symbol") or {},
            title="PnL by symbol",
            ylabel="PnL, USD",
            filename="pnl_by_symbol.png",
        )
        charts["pnl_by_strategy"] = self._bar(
            data=metrics.get("pnl_by_strategy") or {},
            title="PnL by strategy",
            ylabel="PnL, USD",
            filename="pnl_by_strategy.png",
        )
        charts["pnl_by_timeframe"] = self._bar(
            data=metrics.get("pnl_by_timeframe") or {},
            title="PnL by timeframe",
            ylabel="PnL, USD",
            filename="pnl_by_timeframe.png",
        )
        charts["trades_by_symbol"] = self._bar(
            data=metrics.get("trades_by_symbol") or {},
            title="Trades by symbol",
            ylabel="Trades",
            filename="trades_by_symbol.png",
        )
        charts["signals_by_strategy"] = self._bar(
            data=metrics.get("generated_by_strategy") or {},
            title="Generated signals by strategy",
            ylabel="Signals",
            filename="signals_by_strategy.png",
        )
        charts["confirmed_by_strategy"] = self._bar(
            data=metrics.get("confirmed_by_strategy") or {},
            title="Confirmed signals by strategy",
            ylabel="Signals",
            filename="confirmed_by_strategy.png",
        )
        charts["risk_blocks"] = self._bar(
            data=metrics.get("risk_block_reasons") or {},
            title="Risk blocks by reason",
            ylabel="Count",
            filename="risk_blocks.png",
        )
        charts["event_funnel"] = self._bar(
            data=self._event_funnel(metrics.get("event_counters") or {}),
            title="Pipeline event funnel",
            ylabel="Events",
            filename="event_funnel.png",
        )
        charts["event_groups"] = self._bar(
            data=self._event_groups(metrics.get("event_counters") or {}),
            title="Event volume by group",
            ylabel="Events",
            filename="event_groups.png",
        )
        charts["order_topics"] = self._bar(
            data=self._count_by_topic(orders),
            title="Order events",
            ylabel="Events",
            filename="order_topics.png",
        )
        charts["fee_slippage"] = self._bar(
            data={
                "fees": metrics.get("total_fees") or 0,
                "slippage": metrics.get("total_slippage") or 0,
            },
            title="Execution cost",
            ylabel="USD",
            filename="fee_slippage.png",
        )

        return charts

    def _line(self, *, rows: list[dict[str, Any]], y_key: str, title: str, ylabel: str, filename: str) -> str:
        path = self._charts_dir / filename
        fig = plt.figure()
        ax = fig.add_subplot(111)
        values = [safe_float(row.get(y_key)) for row in rows]
        if not values:
            values = [0.0]
        ax.plot(list(range(len(values))), values)
        ax.set_title(title)
        ax.set_xlabel("event")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return f"charts/{filename}"

    def _bar(self, *, data: dict[str, Any], title: str, ylabel: str, filename: str) -> str:
        path = self._charts_dir / filename
        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = [str(k) for k in data.keys()] or ["none"]
        values = [safe_float(v) for v in data.values()] or [0.0]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return f"charts/{filename}"

    def _histogram(self, *, values: list[float], title: str, xlabel: str, filename: str) -> str:
        path = self._charts_dir / filename
        fig = plt.figure()
        ax = fig.add_subplot(111)
        values = values or [0.0]
        ax.hist(values, bins=min(20, max(5, len(values))))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return f"charts/{filename}"

    def _trade_pnl(self, trades: list[dict[str, Any]]) -> str:
        path = self._charts_dir / "trade_pnl.png"
        values = [safe_float(t.get("realized_pnl") or t.get("net_pnl")) for t in trades] or [0.0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar([str(i + 1) for i in range(len(values))], values)
        ax.set_title("Trade PnL by trade")
        ax.set_xlabel("trade #")
        ax.set_ylabel("PnL, USD")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return "charts/trade_pnl.png"

    def _cumulative_pnl(self, trades: list[dict[str, Any]]) -> str:
        path = self._charts_dir / "cumulative_pnl.png"
        values = []
        total = 0.0
        for trade in trades:
            total += safe_float(trade.get("realized_pnl") or trade.get("net_pnl"))
            values.append(total)
        if not values:
            values = [0.0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(list(range(len(values))), values)
        ax.set_title("Cumulative realized PnL")
        ax.set_xlabel("closed trade")
        ax.set_ylabel("PnL, USD")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return "charts/cumulative_pnl.png"

    @staticmethod
    def _event_funnel(counters: dict[str, Any]) -> dict[str, float]:
        return {
            "analytics": sum(safe_float(v) for k, v in counters.items() if str(k).startswith("analytics.")),
            "generated": safe_float(counters.get("signal.generated")),
            "confirmed": safe_float(counters.get("signal.confirmed")),
            "blocked": safe_float(counters.get("risk.position_blocked")),
            "filled": safe_float(counters.get("execution.order_filled")),
            "closed": safe_float(counters.get("position.closed")),
        }

    @staticmethod
    def _event_groups(counters: dict[str, Any]) -> dict[str, float]:
        groups: dict[str, float] = {}
        for topic, value in counters.items():
            group = str(topic).split(".", 1)[0]
            groups[group] = groups.get(group, 0.0) + safe_float(value)
        return groups

    @staticmethod
    def _count_by_topic(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            topic = str(row.get("topic") or "unknown")
            counts[topic] = counts.get(topic, 0) + 1
        return counts

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): BacktestReportBuilder._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [BacktestReportBuilder._json_safe(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    @staticmethod
    def _markdown(*, metrics: dict[str, Any], metadata: dict[str, Any], charts: dict[str, str]) -> str:
        lines = ["# Backtest report", "", "## Metadata"]
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")

        lines.extend(["", "## Summary"])
        for key in (
            "initial_balance",
            "final_balance",
            "net_pnl",
            "total_return_pct",
            "gross_profit",
            "gross_loss",
            "total_fees",
            "total_slippage",
            "closed_trades",
            "winning_trades",
            "losing_trades",
            "breakeven_trades",
            "winrate",
            "profit_factor",
            "expectancy",
            "average_win",
            "average_loss",
            "max_drawdown",
            "max_drawdown_pct",
            "avg_holding_ms",
        ):
            lines.append(f"- **{key}**: {metrics.get(key)}")

        if metrics.get("data_warnings"):
            lines.extend(["", "## Data warnings"])
            for warning in metrics["data_warnings"]:
                lines.append(f"- {warning}")

        lines.extend(["", "## Charts"])
        for name, path in charts.items():
            lines.append(f"- **{name}**: `{path}`")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _html(*, metrics: dict[str, Any], metadata: dict[str, Any], charts: dict[str, str]) -> str:
        def rows(data: dict[str, Any]) -> str:
            return "\n".join(
                f"<tr><th>{html.escape(str(k))}</th><td><pre>{html.escape(str(v))}</pre></td></tr>"
                for k, v in data.items()
            )

        chart_sections = "\n".join(
            f"<section><h2>{html.escape(name.replace('_', ' ').title())}</h2>"
            f"<img src='{html.escape(path)}' alt='{html.escape(name)}'></section>"
            for name, path in charts.items()
        )

        warnings = metrics.get("data_warnings") or []
        warning_html = ""
        if warnings:
            warning_html = "<h2>Data warnings</h2><ul>" + "".join(
                f"<li>{html.escape(str(w))}</li>" for w in warnings
            ) + "</ul>"

        return f"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Backtest Report</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
th {{ width: 280px; background: #f7f7f7; }}
pre {{ white-space: pre-wrap; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }}
section {{ margin: 28px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<h2>Metadata</h2>
<table>{rows(metadata)}</table>
<h2>Summary metrics</h2>
<table>{rows(metrics)}</table>
{warning_html}
<div class="grid">
{chart_sections}
</div>
</body>
</html>
"""
