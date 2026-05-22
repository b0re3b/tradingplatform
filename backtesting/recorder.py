from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from core.event_bus import Event, EventBus, Subscription
from core.logger import get_logger

from backtesting.utils import decimal_from, ensure_dir, pct


@dataclass(slots=True)
class BacktestRecorderStats:
    event_counters: Counter[str] = field(default_factory=Counter)
    risk_block_reasons: Counter[str] = field(default_factory=Counter)
    generated_by_strategy: Counter[str] = field(default_factory=Counter)
    confirmed_by_strategy: Counter[str] = field(default_factory=Counter)
    trades_by_symbol: Counter[str] = field(default_factory=Counter)
    trades_by_strategy: Counter[str] = field(default_factory=Counter)
    pnl_by_symbol: defaultdict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    pnl_by_strategy: defaultdict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    pnl_by_timeframe: defaultdict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))


class BacktestRecorder:
    """Read-only EventBus observer for backtest metrics and detailed report tables."""

    TOPICS: tuple[str, ...] = (
        "analytics.*",
        "signal.generated",
        "signal.confirmed",
        "risk.position_blocked",
        "risk.limit_warning",
        "risk.size_adjusted",
        "execution.order_submitted",
        "execution.order_filled",
        "execution.order_rejected",
        "execution.order_failed",
        "execution.order_cancelled",
        "position.opened",
        "position.updated",
        "position.closed",
        "portfolio.updated",
        "risk.kill_switch",
        "signal.rejected",
        "strategy.engine.batch_processed",
    )

    def __init__(self, *, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._subscriptions: list[Subscription] = []
        self._logger = get_logger(
            __name__,
            service="backtesting.recorder",
            event_type="backtest_recorder",
        )

        self.stats = BacktestRecorderStats()
        self.signals: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.orders: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.portfolio: list[dict[str, Any]] = []
        self.analytics_events: list[dict[str, Any]] = []
        self.rejected_by_reason: dict[str, int] = {}
        self.rejected_by_stage: dict[str, int] = {}
        self.batch_reasons: dict[str, int] = {}
        self.selected_strategies: dict[str, int] = {}

    def register(self) -> None:
        if self._subscriptions:
            return
        for topic in self.TOPICS:
            self._subscriptions.append(
                self._event_bus.subscribe(topic, self._on_event, name=f"backtest_recorder_{topic}")
            )

    def unregister(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription)
        self._subscriptions.clear()

    async def start(self) -> None:
        self.register()

    async def stop(self) -> None:
        self.unregister()

    async def _on_event(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        self.stats.event_counters[event.topic] += 1

        row = {"topic": event.topic, "event_timestamp": event.timestamp, **payload}

        if event.topic.startswith("analytics."):
            self.analytics_events.append(row)

        elif event.topic == "signal.generated":
            self.signals.append(row)
            self.stats.generated_by_strategy[str(payload.get("strategy_name") or "unknown")] += 1

        elif event.topic == "signal.confirmed":
            self.signals.append(row)
            self.stats.confirmed_by_strategy[str(payload.get("strategy_name") or "unknown")] += 1

        elif event.topic == "risk.position_blocked":
            self.signals.append(row)
            self.stats.risk_block_reasons[str(payload.get("reason") or payload.get("decision") or "unknown")] += 1

        elif event.topic.startswith("execution.order_"):
            self.orders.append(row)

        elif event.topic == "position.closed":
            self.trades.append(row)
            symbol = str(payload.get("symbol") or "unknown")
            strategy = str(payload.get("strategy_name") or "unknown")
            timeframe = str(payload.get("timeframe") or "unknown")
            pnl = decimal_from(payload.get("realized_pnl") or payload.get("net_pnl"))
            self.stats.trades_by_symbol[symbol] += 1
            self.stats.trades_by_strategy[strategy] += 1
            self.stats.pnl_by_symbol[symbol] += pnl
            self.stats.pnl_by_strategy[strategy] += pnl
            self.stats.pnl_by_timeframe[timeframe] += pnl

        elif event.topic in {"position.opened", "position.updated"}:
            self.positions.append(row)

        elif event.topic == "portfolio.updated":
            self.portfolio.append(row)

    @property
    def equity_series(self) -> list[dict[str, Any]]:
        if self.portfolio:
            return [
                {
                    "timestamp_ms": int(row.get("timestamp_ms") or 0),
                    "equity": float(row.get("equity") or 0.0),
                    "balance": float(row.get("balance") or 0.0),
                }
                for row in self.portfolio
            ]
        return []

    @property
    def drawdown_series(self) -> list[dict[str, Any]]:
        series = self.equity_series
        peak: float | None = None
        rows: list[dict[str, Any]] = []
        for row in series:
            equity = float(row.get("equity") or 0.0)
            peak = equity if peak is None else max(peak, equity)
            drawdown = equity - peak
            drawdown_pct = 0.0 if not peak else drawdown / peak * 100.0
            rows.append(
                {
                    "timestamp_ms": row.get("timestamp_ms"),
                    "drawdown": drawdown,
                    "drawdown_pct": drawdown_pct,
                }
            )
        return rows

    def metrics(
        self,
        *,
        initial_balance: Decimal,
        final_balance: Decimal,
        total_fees: Decimal,
        total_slippage: Decimal,
        data_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        closed = len(self.trades)
        pnl_values = [decimal_from(t.get("realized_pnl") or t.get("net_pnl")) for t in self.trades]
        wins = sum(1 for pnl in pnl_values if pnl > 0)
        losses = sum(1 for pnl in pnl_values if pnl < 0)
        breakeven = closed - wins - losses
        gross_profit = sum((pnl for pnl in pnl_values if pnl > 0), Decimal("0"))
        gross_loss = sum((pnl for pnl in pnl_values if pnl < 0), Decimal("0"))
        net_pnl = final_balance - initial_balance
        average_win = gross_profit / Decimal(wins) if wins else Decimal("0")
        average_loss = gross_loss / Decimal(losses) if losses else Decimal("0")
        expectancy = (sum(pnl_values, Decimal("0")) / Decimal(closed)) if closed else Decimal("0")
        profit_factor = (
            float(gross_profit / abs(gross_loss))
            if gross_loss < 0
            else None
        )

        holding_times = [int(t.get("holding_ms") or 0) for t in self.trades]
        avg_holding_ms = int(sum(holding_times) / len(holding_times)) if holding_times else 0

        return {
            "initial_balance": float(initial_balance),
            "final_balance": float(final_balance),
            "net_pnl": float(net_pnl),
            "total_return_pct": float(pct(net_pnl, initial_balance)),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "total_fees": float(total_fees),
            "total_slippage": float(total_slippage),
            "closed_trades": closed,
            "winning_trades": wins,
            "losing_trades": losses,
            "breakeven_trades": breakeven,
            "winrate": float(pct(Decimal(wins), Decimal(closed))) if closed else 0.0,
            "loss_rate": float(pct(Decimal(losses), Decimal(closed))) if closed else 0.0,
            "profit_factor": profit_factor,
            "average_win": float(average_win),
            "average_loss": float(average_loss),
            "expectancy": float(expectancy),
            "avg_holding_ms": avg_holding_ms,
            "max_drawdown": min((float(row["drawdown"]) for row in self.drawdown_series), default=0.0),
            "max_drawdown_pct": min((float(row["drawdown_pct"]) for row in self.drawdown_series), default=0.0),
            "event_counters": dict(self.stats.event_counters),
            "risk_block_reasons": dict(self.stats.risk_block_reasons),
            "generated_by_strategy": dict(self.stats.generated_by_strategy),
            "confirmed_by_strategy": dict(self.stats.confirmed_by_strategy),
            "trades_by_symbol": dict(self.stats.trades_by_symbol),
            "trades_by_strategy": dict(self.stats.trades_by_strategy),
            "pnl_by_symbol": {k: float(v) for k, v in self.stats.pnl_by_symbol.items()},
            "pnl_by_strategy": {k: float(v) for k, v in self.stats.pnl_by_strategy.items()},
            "pnl_by_timeframe": {k: float(v) for k, v in self.stats.pnl_by_timeframe.items()},
            "data_warnings": list(data_warnings or []),
        }

    def write_tables(self, output_dir: Path) -> None:
        tables_dir = ensure_dir(output_dir / "tables")
        self._write_csv(tables_dir / "signals.csv", self.signals)
        self._write_csv(tables_dir / "trades.csv", self.trades)
        self._write_csv(tables_dir / "orders.csv", self.orders)
        self._write_csv(tables_dir / "positions.csv", self.positions)
        self._write_csv(tables_dir / "portfolio.csv", self.portfolio)
        self._write_csv(tables_dir / "analytics_events.csv", self.analytics_events)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
