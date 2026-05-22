from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from backtesting.utils import ensure_dir


def _save_bar(data: dict[str, float], *, title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = list(data.keys()) or ["none"]
    values = list(data.values()) or [0.0]
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_line(rows: list[dict[str, Any]], *, y_key: str, title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = [float(row.get(y_key) or 0.0) for row in rows]
    x = list(range(len(y)))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("event")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_charts(*, output_dir: Path, metrics: dict[str, Any], equity_series: list[dict[str, Any]], drawdown_series: list[dict[str, Any]]) -> dict[str, str]:
    ensure_dir(output_dir)
    charts: dict[str, str] = {}

    paths = {
        "equity_curve": output_dir / "equity_curve.png",
        "drawdown_curve": output_dir / "drawdown_curve.png",
        "pnl_by_symbol": output_dir / "pnl_by_symbol.png",
        "pnl_by_strategy": output_dir / "pnl_by_strategy.png",
        "signal_funnel": output_dir / "signal_funnel.png",
    }

    _save_line(equity_series, y_key="equity_delta", title="Equity delta over closed/updated positions", ylabel="USD", path=paths["equity_curve"])
    _save_line(drawdown_series, y_key="drawdown", title="Drawdown curve", ylabel="USD", path=paths["drawdown_curve"])
    _save_bar(metrics.get("pnl_by_symbol") or {}, title="PnL by symbol", ylabel="USD", path=paths["pnl_by_symbol"])
    _save_bar(metrics.get("pnl_by_strategy") or {}, title="PnL by strategy", ylabel="USD", path=paths["pnl_by_strategy"])

    counters = metrics.get("event_counters") or {}
    funnel = {
        "analytics": sum(v for k, v in counters.items() if str(k).startswith("analytics.")),
        "signals": counters.get("signal.generated", 0),
        "confirmed": counters.get("signal.confirmed", 0),
        "filled": counters.get("execution.order_filled", 0),
        "winners": metrics.get("winning_trades", 0),
    }
    _save_bar(funnel, title="Signal funnel", ylabel="count", path=paths["signal_funnel"])

    for key, path in paths.items():
        charts[key] = path.name
    return charts
