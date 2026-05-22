from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger, init_logger
from core.scheduler import Scheduler

from backtesting.binance_history import BinanceHistoryLoader
from backtesting.config import BacktestConfig
from backtesting.factory import BacktestFactory
from backtesting.paper_execution import BacktestPaperExecution
from backtesting.recorder import BacktestRecorder
from backtesting.replay import HistoricalMarketReplay
from backtesting.report import BacktestReportBuilder
from backtesting.utils import (
    assert_backtest_safety,
    default_period,
    drain_event_bus,
    ensure_dir,
    ms_to_datetime,
)


@dataclass(slots=True)
class BacktestResult:
    backtest_id: str
    report_path: Path
    metrics: dict[str, Any]
    replayed_events: int


class BacktestEngine:
    """Orchestrates production-component lifecycle, historical loading, replay and reporting."""

    def __init__(self, *, config: BacktestConfig, factory: BacktestFactory) -> None:
        self._config = config
        self._config.validate()
        self._factory = factory
        self._logger = get_logger(
            __name__,
            service="backtesting.engine",
            event_type="backtest_engine",
        )

    async def run(self) -> BacktestResult:
        assert_backtest_safety(env_guard=self._config.backtest_mode_env_guard)

        backtest_id = self._config.backtest_id or self._make_backtest_id()
        output_dir = ensure_dir(self._config.reports_dir / backtest_id)

        init_logger()

        core_config = Config.from_env() if hasattr(Config, "from_env") else Config()
        core_config.exchange.name = self._config.exchange
        core_config.exchange.rest_url = self._config.binance_public_base_url

        event_bus = self._build_event_bus(core_config)
        scheduler = Scheduler(
            event_bus=event_bus,
            tick_interval=getattr(core_config.scheduler, "tick_interval", 0.2),
            service_name="backtesting.scheduler",
        )

        paper_execution = BacktestPaperExecution(config=self._config, event_bus=event_bus)
        recorder = BacktestRecorder(event_bus=event_bus)

        active_services: list[Any] = []
        data_warnings: list[str] = []
        replayed_events = 0
        scheduler_started = False
        start_ms = 0
        end_ms = 0
        report_path: Path | None = None

        try:
            caches = await self._factory.build_caches(
                config=core_config,
                event_bus=event_bus,
                scheduler=scheduler,
            )
            analytics = await self._factory.build_analytics(
                config=core_config,
                event_bus=event_bus,
                scheduler=scheduler,
                caches=caches,
            )
            strategy = await self._factory.build_strategy(
                config=core_config,
                event_bus=event_bus,
                scheduler=scheduler,
            )
            risk = await self._factory.build_risk(
                config=core_config,
                event_bus=event_bus,
                scheduler=scheduler,
                initial_balance=float(self._config.initial_balance_usd),
            )

            await event_bus.start()

            if self._config.disable_scheduler_loop:
                self._logger.info(
                    "Scheduler loop disabled for deterministic backtest; "
                    "components may register jobs but they will not run."
                )
            else:
                await scheduler.start()
                scheduler_started = True

            services = self._dedupe_services(
                [
                    *caches,
                    *analytics,
                    *strategy,
                    risk,
                    paper_execution,
                    recorder,
                ]
            )

            for service in services:
                await self._activate_service(service)
                active_services.append(service)

            start_ms, end_ms = default_period(
                self._config.lookback_days,
                smallest_timeframe=min(self._config.timeframes, key=self._timeframe_rank),
            )

            async with BinanceHistoryLoader(self._config) as loader:
                dataset = await loader.load(start_time_ms=start_ms, end_time_ms=end_ms)
                data_warnings = dataset.warnings

            replay = HistoricalMarketReplay(config=self._config, event_bus=event_bus)
            events = replay.build_events(dataset, backtest_id=backtest_id)
            replayed_events = await replay.replay(events)

            await self._drain_nonfatal(event_bus, label="post_replay")

            await paper_execution.close_all_at_last_price(reason="backtest_end")

            await self._drain_nonfatal(event_bus, label="post_close_all")

            report_path, metrics = self._write_report(
                output_dir=output_dir,
                backtest_id=backtest_id,
                start_ms=start_ms,
                end_ms=end_ms,
                replayed_events=replayed_events,
                recorder=recorder,
                paper_execution=paper_execution,
                data_warnings=data_warnings,
                status="completed",
            )

            return BacktestResult(
                backtest_id=backtest_id,
                report_path=report_path,
                metrics=metrics,
                replayed_events=replayed_events,
            )

        except BaseException as exc:
            # Best-effort partial report. This is useful when replay is interrupted,
            # a handler hangs, or a downstream service fails before normal report build.
            try:
                report_path, metrics = self._write_report(
                    output_dir=output_dir,
                    backtest_id=backtest_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    replayed_events=replayed_events,
                    recorder=recorder,
                    paper_execution=paper_execution,
                    data_warnings=[*data_warnings, f"Backtest interrupted/failed: {type(exc).__name__}: {exc}"],
                    status="partial_failed",
                )
                self._logger.warning(
                    "Partial backtest report written after failure | path=%s error=%s",
                    report_path,
                    exc,
                )
            except Exception as report_exc:
                self._logger.exception(
                    "Failed to write partial backtest report | report_error=%s original_error=%s",
                    report_exc,
                    exc,
                )
            raise

        finally:
            for service in reversed(active_services):
                await self._stop_service_safe(service)

            if scheduler_started:
                await self._stop_scheduler_safe(scheduler)

            await self._stop_event_bus_safe(event_bus)

    async def _drain_nonfatal(self, event_bus: EventBus, *, label: str) -> None:
        drained = await drain_event_bus(
            event_bus,
            require_public_join=self._config.require_event_bus_join,
            timeout=float(self._config.replay_final_drain_timeout_seconds),
            raise_on_timeout=False,
        )
        if not drained:
            self._logger.warning(
                "Backtest drain timed out; continuing to report build | stage=%s",
                label,
            )

    def _write_report(
        self,
        *,
        output_dir: Path,
        backtest_id: str,
        start_ms: int,
        end_ms: int,
        replayed_events: int,
        recorder: BacktestRecorder,
        paper_execution: BacktestPaperExecution,
        data_warnings: list[str],
        status: str,
    ) -> tuple[Path, dict[str, Any]]:
        self._logger.info(
            "Building backtest report | status=%s output_dir=%s replayed_events=%s",
            status,
            output_dir,
            replayed_events,
        )

        metrics = recorder.metrics(
            initial_balance=self._config.initial_balance_usd,
            final_balance=paper_execution.balance,
            total_fees=paper_execution.total_fees,
            total_slippage=paper_execution.total_slippage,
            data_warnings=data_warnings,
        )
        recorder.write_tables(output_dir)

        period = "unknown"
        if start_ms and end_ms:
            period = f"{ms_to_datetime(start_ms).isoformat()} → {ms_to_datetime(end_ms).isoformat()}"

        report_path = BacktestReportBuilder(
            output_dir=output_dir,
            report_format=self._config.report_format,
        ).build(
            metrics=metrics,
            equity_series=recorder.equity_series,
            drawdown_series=recorder.drawdown_series,
            trades=recorder.trades,
            orders=recorder.orders,
            signals=recorder.signals,
            portfolio=recorder.portfolio,
            metadata={
                "backtest_id": backtest_id,
                "status": status,
                "period": period,
                "exchange": self._config.exchange,
                "market_type": self._config.market_type,
                "symbols": ", ".join(self._config.symbols),
                "timeframes": ", ".join(self._config.timeframes),
                "initial_balance": self._config.initial_balance_usd,
                "execution_mode": self._config.execution_mode,
                "replayed_events": replayed_events,
                "agg_trades_enabled": self._config.enable_agg_trades,
                "funding_enabled": self._config.enable_funding,
                "open_interest_enabled": self._config.enable_open_interest,
                "orderflow_enabled": self._config.enable_orderflow,
                "scheduler_loop_disabled": self._config.disable_scheduler_loop,
            },
        )

        self._logger.info("Backtest report written | path=%s", report_path)
        return report_path, metrics

    def _build_event_bus(self, core_config: Config) -> EventBus:
        kwargs = {
            "max_queue_size": max(
                int(getattr(core_config.event_bus, "max_queue_size", 20_000)),
                self._config.event_bus_max_queue_size,
            ),
            "worker_count": max(
                int(getattr(core_config.event_bus, "worker_count", 6)),
                self._config.event_bus_worker_count,
            ),
            "max_retries": getattr(core_config.event_bus, "max_retries", 3),
            "retry_delay": getattr(core_config.event_bus, "retry_delay", 0.5),
        }

        try:
            event_bus = EventBus(**kwargs, overflow_policy=self._config.event_bus_overflow_policy)
        except TypeError:
            event_bus = EventBus(**kwargs)

        self._force_event_bus_policy(event_bus)
        return event_bus

    def _force_event_bus_policy(self, event_bus: EventBus) -> None:
        for attr in (
            "_overflow_policy",
            "overflow_policy",
            "_queue_overflow_policy",
            "queue_overflow_policy",
            "_policy",
            "policy",
        ):
            if hasattr(event_bus, attr):
                try:
                    setattr(event_bus, attr, self._config.event_bus_overflow_policy)
                except Exception:
                    pass

    def _dedupe_services(self, services: list[Any]) -> list[Any]:
        seen: set[int] = set()
        result: list[Any] = []
        for service in services:
            if service is None:
                continue
            ident = id(service)
            if ident in seen:
                self._logger.warning(
                    "Skipped duplicate service instance | service=%s",
                    service.__class__.__name__,
                )
                continue
            seen.add(ident)
            result.append(service)
        return result

    async def _activate_service(self, service: Any) -> None:
        if getattr(service, "_backtest_active", False):
            return

        if self._is_register_only(service):
            await self._call_if_exists(service, "register")
        else:
            if hasattr(service, "start"):
                await self._call_if_exists(service, "start")
            else:
                await self._call_if_exists(service, "register")

        setattr(service, "_backtest_active", True)

    @staticmethod
    def _is_register_only(service: Any) -> bool:
        cls_name = service.__class__.__name__
        return cls_name.endswith("Cache") or cls_name in {
            "BacktestRecorder",
            "BacktestPaperExecution",
        }

    async def _stop_service_safe(self, service: Any) -> None:
        try:
            if self._is_register_only(service):
                await self._call_if_exists(service, "unregister")
            else:
                await self._call_if_exists(service, "stop")
        except Exception as exc:
            self._logger.warning(
                "Service stop failed during backtest cleanup | service=%s error=%s",
                service.__class__.__name__,
                exc,
            )

    async def _stop_scheduler_safe(self, scheduler: Scheduler) -> None:
        try:
            await scheduler.stop()
        except Exception as exc:
            self._logger.warning("Scheduler stop failed during backtest cleanup | error=%s", exc)

    async def _stop_event_bus_safe(self, event_bus: EventBus) -> None:
        stop = getattr(event_bus, "stop", None)
        if not callable(stop):
            return

        try:
            # Avoid hanging after report generation. At this point report is already written.
            result = stop(drain=False)
        except TypeError:
            result = stop()

        if inspect.isawaitable(result):
            try:
                await result
            except Exception as exc:
                self._logger.warning("EventBus stop failed during backtest cleanup | error=%s", exc)

    @staticmethod
    async def _call_if_exists(service: Any, method_name: str) -> None:
        method = getattr(service, method_name, None)
        if not callable(method):
            return
        result = method()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _timeframe_rank(timeframe: str) -> int:
        order = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        return order.get(timeframe, 10_000)

    def _make_backtest_id(self) -> str:
        from datetime import datetime, timezone

        stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        symbols = "_".join(s.replace("USDT", "").lower() for s in self._config.symbols)
        return f"{stamp}_{self._config.lookback_days}d_{self._config.exchange}_{symbols}"
