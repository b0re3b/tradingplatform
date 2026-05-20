"""
Full-pipeline strategy tester for backtesting.

StrategyTester is the main offline orchestrator.

It runs the system as close as possible to production flow:

    BacktestDataset
        -> MarketReplay
        -> core.EventBus market.*
        -> production data caches
        -> production analytics
        -> StrategyEngine / SignalProcessor
        -> signal.generated
        -> RiskManager
        -> signal.confirmed / risk.position_blocked
        -> ExecutionSimulator
        -> execution.order_*
        -> PositionSimulator
        -> position.*
        -> PerformanceMetrics / ModelAnalytics / ReportBuilder

Important:
- StrategyTester does not call individual strategies directly.
- StrategyTester does not bypass RiskManager.
- StrategyTester does not open positions directly.
- StrategyTester does not use live exchange execution.
- Live execution is replaced only by ExecutionSimulator + PositionSimulator.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any

from core.event_bus import EventBus
from core.logger import get_logger

from backtesting.backtest_time import BacktestClock
from backtesting.config import BacktestConfig, StrategyTesterConfig
from backtesting.cost_models import TradingCostModel
from backtesting.enums import (
    BacktestMode,
    BacktestStatus,
    BacktestWarningLevel,
    SignalOutcome,
)
from backtesting.exceptions import (
    BacktestDependencyError,
    BacktestLifecycleError,
    BacktestResultCollectionError,
    StrategyBacktestRunError,
    StrategyRegistryEmptyError,
    StrategySelectionError,
    wrap_backtest_error,
)
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.market_replay import MarketReplay
from backtesting.model_analytics import BacktestModelAnalyticsEngine, ModelAnalyticsInput
from backtesting.models import (
    BacktestDataset,
    BacktestEvent,
    BacktestExecutionRecord,
    BacktestPositionRecord,
    BacktestResult,
    BacktestRiskDecisionRecord,
    BacktestSignalRecord,
    SimulationModelSnapshot,
    new_id,
    timestamp_ms,
    utcnow,
)
from backtesting.performance_metrics import (
    PerformanceMetrics,
    build_metrics_input_from_components,
)
from backtesting.position_simulator import PositionSimulator
from backtesting.report_builder import ReportBuilder


@dataclass(slots=True)
class BacktestComponentBundle:
    """
    Runtime component bundle used by StrategyTester.

    Production components are injected from outside:
    - data caches;
    - analytics components;
    - StrategyEngine / SignalProcessor;
    - RiskManager.

    Backtesting-only components may be built by StrategyTester:
    - MarketReplay;
    - ExecutionSimulator;
    - PositionSimulator;
    - PerformanceMetrics;
    - ModelAnalytics;
    - ReportBuilder;
    - BacktestClock;
    - TradingCostModel.
    """

    event_bus: EventBus
    scheduler: Any | None = None

    data_caches: list[Any] = field(default_factory=list)
    analytics_components: list[Any] = field(default_factory=list)

    strategy_registry: Any | None = None
    strategy_engine: Any | None = None
    signal_processor: Any | None = None

    risk_manager: Any | None = None

    market_replay: MarketReplay | None = None
    execution_simulator: ExecutionSimulator | None = None
    position_simulator: PositionSimulator | None = None

    performance_metrics: PerformanceMetrics | None = None
    model_analytics: BacktestModelAnalyticsEngine | None = None
    report_builder: ReportBuilder | None = None

    cost_model: TradingCostModel | None = None
    clock: BacktestClock | None = None


@dataclass(slots=True)
class BacktestCollectors:
    """
    Event collectors for audit and final result construction.
    """

    events: list[BacktestEvent] = field(default_factory=list)
    signals: list[BacktestSignalRecord] = field(default_factory=list)
    risk_decisions: list[BacktestRiskDecisionRecord] = field(default_factory=list)
    execution_records: list[BacktestExecutionRecord] = field(default_factory=list)
    position_records: list[BacktestPositionRecord] = field(default_factory=list)

    def reset(self) -> None:
        self.events.clear()
        self.signals.clear()
        self.risk_decisions.clear()
        self.execution_records.clear()
        self.position_records.clear()


class StrategyTester:
    """
    Main full-pipeline backtest runner.

    This class is intentionally an orchestrator only. It wires historical replay
    into the real EventBus-driven production pipeline and replaces only live
    execution/position accounting with backtesting simulators.
    """

    def __init__(
        self,
        config: BacktestConfig | StrategyTesterConfig,
        *,
        dataset: BacktestDataset | None = None,
        event_bus: EventBus | None = None,
        scheduler: Any | None = None,
        data_caches: list[Any] | None = None,
        analytics_components: list[Any] | None = None,
        strategy_registry: Any | None = None,
        strategy_engine: Any | None = None,
        signal_processor: Any | None = None,
        risk_manager: Any | None = None,
        market_replay: MarketReplay | None = None,
        execution_simulator: ExecutionSimulator | None = None,
        position_simulator: PositionSimulator | None = None,
        performance_metrics: PerformanceMetrics | None = None,
        model_analytics: BacktestModelAnalyticsEngine | None = None,
        report_builder: ReportBuilder | None = None,
        cost_model: TradingCostModel | None = None,
        clock: BacktestClock | None = None,
        logger_name: str = "backtesting.strategy_tester",
    ) -> None:
        if isinstance(config, BacktestConfig):
            config.validate()
            self.backtest_config: BacktestConfig | None = config
            self.config = config.strategy_tester
        else:
            config.validate()
            self.backtest_config = None
            self.config = config

        self.dataset = dataset
        self.logger = get_logger(logger_name)

        # Full-pipeline backtesting must use the real production EventBus.
        # If one is injected by run.py, that exact instance is used.
        # If omitted, StrategyTester creates a real EventBus, not a lightweight
        # in-memory substitute.
        resolved_event_bus = event_bus if event_bus is not None else EventBus()

        self.components = BacktestComponentBundle(
            event_bus=resolved_event_bus,
            scheduler=scheduler,
            data_caches=data_caches or [],
            analytics_components=analytics_components or [],
            strategy_registry=strategy_registry,
            strategy_engine=strategy_engine,
            signal_processor=signal_processor,
            risk_manager=risk_manager,
            market_replay=market_replay,
            execution_simulator=execution_simulator,
            position_simulator=position_simulator,
            performance_metrics=performance_metrics,
            model_analytics=model_analytics,
            report_builder=report_builder,
            cost_model=cost_model,
            clock=clock,
        )

        self.collectors = BacktestCollectors()
        self.result: BacktestResult | None = None

        self._collector_subscriptions: list[Any] = []
        self._registered = False
        self._prepared = False
        self._running = False
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Register event collectors and runtime components.

        Registration order matters:
        - collectors first, so diagnostics are not missed;
        - data caches before analytics;
        - analytics before strategy;
        - strategy before risk;
        - risk before execution/position simulation.
        """

        if self._registered:
            return

        self._register_collectors()

        for component in self._all_registerable_components():
            register = getattr(component, "register", None)
            if callable(register):
                register()

        self._registered = True

        self.logger.info(
            "StrategyTester registered",
            extra={
                "event_bus_type": self.components.event_bus.__class__.__name__,
                "data_caches": [item.__class__.__name__ for item in self.components.data_caches],
                "analytics_components": [
                    item.__class__.__name__ for item in self.components.analytics_components
                ],
                "has_strategy_engine": self.components.strategy_engine is not None,
                "has_signal_processor": self.components.signal_processor is not None,
                "has_risk_manager": self.components.risk_manager is not None,
                "collector_subscriptions": len(self._collector_subscriptions),
            },
        )

    async def prepare_environment(
        self,
        *,
        dataset: BacktestDataset | None = None,
    ) -> None:
        """
        Build and validate the backtest runtime environment.
        """

        async with self._lock:
            if dataset is not None:
                self.dataset = dataset

            if self.dataset is None:
                raise BacktestLifecycleError(
                    "StrategyTester requires BacktestDataset. "
                    "Load data with DataLoader and pass dataset before run()."
                )

            self._validate_dataset()
            self._build_missing_backtesting_components()
            self._validate_pipeline_dependencies()
            self._prepare_market_replay()
            self.register()
            self._prepared = True

    async def _drain_event_bus_after_replay(self) -> None:
        """
        Give real EventBus/Scheduler time to process queued replay events.

        core.EventBus may dispatch through an internal async queue. MarketReplay can
        finish emitting before downstream handlers/counters/data caches have fully
        processed all queued events.
        """

        event_bus = self.components.event_bus

        # Prefer explicit drain/join/wait methods if core EventBus exposes them.
        for method_name in (
                "drain",
                "join",
                "wait_until_idle",
                "wait_empty",
                "flush",
        ):
            method = getattr(event_bus, method_name, None)

            if not callable(method):
                continue

            try:
                result = method()
                if inspect.isawaitable(result):
                    await result
                return
            except TypeError:
                continue

        # Fallback: yield several event-loop ticks so EventBus workers can dispatch.
        for _ in range(10):
            await asyncio.sleep(0)
    async def run(
        self,
        *,
        dataset: BacktestDataset | None = None,
    ) -> BacktestResult:
        """
        Run full backtest.
        """

        if dataset is not None:
            self.dataset = dataset

        if not self._prepared:
            await self.prepare_environment(dataset=self.dataset)

        result = self._create_result()
        self.result = result
        result.mark_started()

        self._running = True

        try:
            await self._start_components()
            await self._run_replay()
            await self._drain_event_bus_after_replay()
            await self._stop_components()

            self._collect_results(result)
            self._calculate_metrics(result)
            self._calculate_model_analytics(result)
            self._build_report(result)

            result.final_balance = self._resolve_final_balance()
            result.final_equity = self._resolve_final_equity()

            result.mark_completed()
            return result

        except Exception as exc:
            wrapped = wrap_backtest_error(
                exc,
                "Strategy backtest run failed.",
                code="strategy_backtest_run_failed",
                details={
                    "run_name": self.config.run_name,
                    "mode": self.config.mode.value,
                    "error_type": exc.__class__.__name__,
                },
            )

            result.mark_failed(
                wrapped.message,
                details=wrapped.to_dict(),
            )

            try:
                await self._stop_components()
            except Exception as stop_exc:
                result.add_warning(
                    "Failed to stop all components after backtest failure.",
                    level=BacktestWarningLevel.ERROR,
                    code="component_stop_failed",
                    details={
                        "error": str(stop_exc),
                        "error_type": stop_exc.__class__.__name__,
                    },
                )

            if self.config.stop_on_first_error:
                raise StrategyBacktestRunError(
                    "Strategy backtest failed.",
                    details=result.error_details,
                ) from exc

            return result

        finally:
            self._running = False

            if self.config.cleanup_after_run:
                await self.cleanup()

    async def run_single_strategy(
        self,
        strategy_name: str,
        *,
        dataset: BacktestDataset | None = None,
    ) -> BacktestResult:
        """
        Run backtest for one selected strategy.
        """

        if not strategy_name:
            raise StrategySelectionError("strategy_name is required.")

        self.config.test_all_registered_strategies = False
        self.config.strategies = [strategy_name]
        return await self.run(dataset=dataset)

    async def run_multi_strategy(
        self,
        strategies: list[str] | None = None,
        *,
        dataset: BacktestDataset | None = None,
    ) -> BacktestResult:
        """
        Run multiple strategies through StrategyRegistry/StrategyEngine.
        """

        self.config.mode = BacktestMode.MULTI_STRATEGY

        if strategies is not None:
            self.config.test_all_registered_strategies = False
            self.config.strategies = list(strategies)

        return await self.run(dataset=dataset)

    async def run_portfolio(
        self,
        *,
        dataset: BacktestDataset | None = None,
    ) -> BacktestResult:
        """
        Run portfolio-level test across configured symbols/strategies.
        """

        self.config.mode = BacktestMode.PORTFOLIO
        return await self.run(dataset=dataset)

    async def cleanup(self) -> None:
        """
        Cleanup volatile runtime flags and collector subscriptions.
        """

        self._prepared = False
        self._running = False

        # Do not stop/destroy production components here. StrategyTester owns
        # only its collector subscriptions and volatile flags.
        self._unregister_collectors()

    # -------------------------------------------------------------------------
    # Component setup
    # -------------------------------------------------------------------------

    def _build_missing_backtesting_components(self) -> None:
        """
        Build missing backtesting-only components.

        Production data/analytics/strategy/risk components are not built here.
        They must be injected by the runner.
        """

        bt_config = self.backtest_config

        if self.components.cost_model is None:
            self.components.cost_model = TradingCostModel(
                bt_config.cost_model if bt_config else None
            )

        if self.components.clock is None:
            if bt_config is not None:
                self.components.clock = BacktestClock(
                    period=bt_config.period(),
                    config=bt_config.backtest_time,
                )
            elif self.dataset and self.dataset.info.period:
                self.components.clock = BacktestClock(self.dataset.info.period)
            else:
                raise BacktestDependencyError(
                    "BacktestClock is missing and cannot be inferred from config/dataset."
                )

        if self.components.market_replay is None:
            self.components.market_replay = MarketReplay(
                config=bt_config.market_replay if bt_config else None,
                event_bus=self.components.event_bus,
                clock=self.components.clock,
            )

        if self.components.execution_simulator is None:
            self.components.execution_simulator = ExecutionSimulator(
                config=bt_config.execution_simulator if bt_config else None,
                event_bus=self.components.event_bus,
                clock=self.components.clock,
                cost_model=self.components.cost_model,
                random_seed=bt_config.random_seed if bt_config else 42,
            )

        if self.components.position_simulator is None:
            self.components.position_simulator = PositionSimulator(
                config=bt_config.position_simulator if bt_config else None,
                event_bus=self.components.event_bus,
                clock=self.components.clock,
                cost_model=self.components.cost_model,
            )

        if self.components.performance_metrics is None:
            self.components.performance_metrics = PerformanceMetrics(
                bt_config.performance_metrics if bt_config else None
            )

        if self.components.model_analytics is None:
            self.components.model_analytics = BacktestModelAnalyticsEngine(
                bt_config.model_analytics if bt_config else None
            )

        if self.components.report_builder is None:
            self.components.report_builder = ReportBuilder(
                bt_config.report_builder if bt_config else None
            )

    def _prepare_market_replay(self) -> None:
        if self.components.market_replay is None:
            raise BacktestDependencyError("MarketReplay is missing.")

        if self.dataset is None:
            raise BacktestLifecycleError("Cannot prepare MarketReplay without dataset.")

        self.components.market_replay.prepare(
            self.dataset,
            clock=self.components.clock,
        )

    def _validate_dataset(self) -> None:
        if self.dataset is None:
            raise BacktestLifecycleError("Dataset is missing.")

        if self.dataset.is_empty:
            raise BacktestLifecycleError("Dataset is empty.")

        if self.dataset.info is not None:
            self.dataset.info.validate()

    def _validate_pipeline_dependencies(self) -> None:
        """
        Validate production components required for full-pipeline backtesting.
        """

        if self.components.event_bus is None:
            raise BacktestDependencyError("EventBus is required for StrategyTester.")

        if not isinstance(self.components.event_bus, EventBus):
            raise BacktestDependencyError(
                "StrategyTester full pipeline requires core.event_bus.EventBus.",
                details={
                    "event_bus_type": self.components.event_bus.__class__.__name__,
                },
            )

        if self.config.require_strategy_engine and self.components.strategy_engine is None:
            raise BacktestDependencyError(
                "StrategyEngine is required. Inject production StrategyEngine."
            )

        if self.config.require_signal_processor and self.components.signal_processor is None:
            if not self._component_has_attr(
                self.components.strategy_engine,
                ["signal_processor", "processor"],
            ):
                raise BacktestDependencyError(
                    "SignalProcessor is required. Inject it or use StrategyEngine that owns it."
                )

        if self.config.require_risk_manager and self.components.risk_manager is None:
            raise BacktestDependencyError(
                "RiskManager is required so backtest cannot bypass risk."
            )

        if self.config.require_analytics and not self.components.analytics_components:
            raise BacktestDependencyError(
                "Analytics components are required. Inject production analytics components."
            )

        if not self.components.data_caches:
            self.logger.warning(
                "No production data caches were injected. "
                "Analytics that listen to market.*.updated will not receive replayed data."
            )

        self._assert_single_event_bus_instance()

        if self.config.fail_if_live_execution_detected:
            self._assert_no_live_execution_component()

        if self.components.strategy_registry is not None:
            self._validate_strategy_selection()

    def _assert_single_event_bus_instance(self) -> None:
        """
        Best-effort validation that injected components use the same EventBus.
        """

        expected = self.components.event_bus
        mismatches: list[dict[str, str]] = []

        for component in self._components_using_event_bus():
            component_bus = self._extract_component_event_bus(component)

            if component_bus is None:
                continue

            if component_bus is not expected:
                mismatches.append(
                    {
                        "component": component.__class__.__name__,
                        "component_event_bus_id": str(id(component_bus)),
                        "expected_event_bus_id": str(id(expected)),
                    }
                )

        if mismatches:
            raise BacktestDependencyError(
                "All backtest pipeline components must share the same EventBus instance.",
                details={"mismatches": mismatches},
            )

    def _components_using_event_bus(self) -> list[Any]:
        components = [
            *self.components.data_caches,
            *self.components.analytics_components,
            self.components.signal_processor,
            self.components.strategy_engine,
            self.components.risk_manager,
            self.components.market_replay,
            self.components.execution_simulator,
            self.components.position_simulator,
        ]
        return [component for component in components if component is not None]

    @staticmethod
    def _extract_component_event_bus(component: Any) -> Any | None:
        for attr in ("event_bus", "_event_bus", "bus", "_bus"):
            value = getattr(component, attr, None)
            if value is not None:
                return value
        return None

    def _validate_strategy_selection(self) -> None:
        registry = self.components.strategy_registry

        if registry is None:
            return

        strategies = self._get_registered_strategies(registry)

        if not strategies:
            raise StrategyRegistryEmptyError("StrategyRegistry has no registered strategies.")

        if self.config.test_all_registered_strategies:
            return

        requested = set(self.config.strategies)

        if not requested and self.config.strategy_preset is None:
            raise StrategySelectionError(
                "No strategies selected and test_all_registered_strategies=False."
            )

        registered_names = {
            self._strategy_name(strategy)
            for strategy in strategies
        }

        missing = sorted(requested - registered_names)

        if missing:
            raise StrategySelectionError(
                "Requested strategies are not registered.",
                details={
                    "missing": missing,
                    "registered": sorted(registered_names),
                },
            )

    def _assert_no_live_execution_component(self) -> None:
        """
        Best-effort guard against accidentally using live execution.
        """

        suspicious_names = {
            "trade_executor",
            "order_manager",
            "binance_rest",
            "bybit_rest",
            "okx_rest",
            "mexc_rest",
            "exchange_client",
            "rest_client",
            "live_exchange",
        }

        owners = {
            "strategy_engine": self.components.strategy_engine,
            "risk_manager": self.components.risk_manager,
            "execution_simulator": self.components.execution_simulator,
            "position_simulator": self.components.position_simulator,
        }

        for owner_name, owner in owners.items():
            if owner is None:
                continue

            for attr_name in suspicious_names:
                value = getattr(owner, attr_name, None)
                if value is None:
                    continue

                if isinstance(value, ExecutionSimulator):
                    continue

                raise BacktestDependencyError(
                    "Potential live execution dependency detected during backtest.",
                    details={
                        "owner": owner_name,
                        "attribute": attr_name,
                        "type": value.__class__.__name__,
                    },
                )

    # -------------------------------------------------------------------------
    # Component lifecycle
    # -------------------------------------------------------------------------

    async def _start_components(self) -> None:
        """
        Start components in pipeline order.

        MarketReplay is prepared earlier but replayed only after all components
        are registered and started.
        """

        if self.components.clock is not None and not self.components.clock.started:
            self.components.clock.start(
                total_events=len(self.dataset.events) if self.dataset else 0
            )

        for component in self._components_start_order():
            await self._maybe_call(component, "start")

    async def _stop_components(self) -> None:
        """
        Stop components in reverse pipeline order.
        """

        for component in self._components_stop_order():
            await self._maybe_call(component, "stop")

        if self.components.clock is not None and self.components.clock.started:
            try:
                self.components.clock.stop()
            except Exception:
                # Clock may already be stopped by MarketReplay.
                pass

    async def _run_replay(self) -> None:
        if self.components.market_replay is None:
            raise BacktestDependencyError("MarketReplay is missing.")

        await self.components.market_replay.replay()

    def _components_start_order(self) -> list[Any]:
        components: list[Any] = []

        components.extend(self.components.data_caches)
        components.extend(self.components.analytics_components)

        components.extend(
            [
                self.components.signal_processor,
                self.components.strategy_engine,
                self.components.risk_manager,
                self.components.execution_simulator,
                self.components.position_simulator,
            ]
        )

        return self._dedupe_components(components)

    def _components_stop_order(self) -> list[Any]:
        components = [
            self.components.position_simulator,
            self.components.execution_simulator,
            self.components.risk_manager,
            self.components.strategy_engine,
            self.components.signal_processor,
            *reversed(self.components.analytics_components),
            *reversed(self.components.data_caches),
        ]
        return self._dedupe_components(components)

    def _all_registerable_components(self) -> list[Any]:
        components = [
            *self.components.data_caches,
            *self.components.analytics_components,
            self.components.signal_processor,
            self.components.strategy_engine,
            self.components.risk_manager,
            self.components.execution_simulator,
            self.components.position_simulator,
            self.components.market_replay,
        ]
        return self._dedupe_components(components)

    @staticmethod
    def _dedupe_components(components: list[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[int] = set()

        for component in components:
            if component is None:
                continue

            key = id(component)
            if key in seen:
                continue

            seen.add(key)
            result.append(component)

        return result

    @staticmethod
    async def _maybe_call(component: Any, method_name: str) -> Any:
        method = getattr(component, method_name, None)

        if not callable(method):
            return None

        result = method()

        if inspect.isawaitable(result):
            return await result

        return result

    # -------------------------------------------------------------------------
    # Collectors
    # -------------------------------------------------------------------------

    def _register_collectors(self) -> None:
        event_bus = self.components.event_bus

        if self.config.collect_signal_records:
            self._subscribe_collector("signal.generated", self._collect_signal_generated)
            self._subscribe_collector("signal.rejected", self._collect_signal_rejected)
            self._subscribe_collector("signal.updated", self._collect_signal_updated)

        if self.config.collect_risk_records:
            self._subscribe_collector("signal.confirmed", self._collect_signal_confirmed)
            self._subscribe_collector("risk.position_blocked", self._collect_risk_blocked)
            self._subscribe_collector("risk.kill_switch", self._collect_risk_kill_switch)
            self._subscribe_collector("risk.limit_warning", self._collect_risk_limit_warning)

        if self.config.collect_execution_records:
            self._subscribe_collector("execution.order_submitted", self._collect_execution_event)
            self._subscribe_collector("execution.order_rejected", self._collect_execution_event)
            self._subscribe_collector("execution.order_failed", self._collect_execution_event)
            self._subscribe_collector("execution.order_cancelled", self._collect_execution_event)
            self._subscribe_collector("execution.order_filled", self._collect_execution_event)
            self._subscribe_collector(
                "execution.order_partially_filled",
                self._collect_execution_event,
            )

        if self.config.collect_position_records:
            self._subscribe_collector("position.opened", self._collect_position_event)
            self._subscribe_collector("position.updated", self._collect_position_event)
            self._subscribe_collector("position.closed", self._collect_position_event)
            self._subscribe_collector("position.liquidated", self._collect_position_event)

        if self.config.collect_event_log:
            # Lightweight pipeline diagnostics. These are intentionally broad.
            # They collect only payload snapshots from key production topics.
            for topic in (
                "market.*",
                "analytics.*",
                "strategy.*",
                "signal.*",
                "risk.*",
                "execution.*",
                "position.*",
                "system.backtest.*",
            ):
                self._subscribe_collector(topic, self._collect_event_log)

        self.logger.debug(
            "StrategyTester collectors registered",
            extra={
                "subscriptions": len(self._collector_subscriptions),
                "event_bus_type": event_bus.__class__.__name__,
            },
        )

    def _subscribe_collector(self, topic: str, handler: Any) -> None:
        """
        Subscribe a collector to production EventBus.

        Collector handlers accept Event or dict. This method uses production
        EventBus API and keeps returned subscriptions so cleanup can unsubscribe.
        """

        subscribe = getattr(self.components.event_bus, "subscribe", None)

        if not callable(subscribe):
            raise BacktestDependencyError("EventBus does not support subscribe().")

        wrapped_handler = self._wrap_collector_handler(topic, handler)

        try:
            subscription = subscribe(
                topic,
                wrapped_handler,
                name=f"backtest_collector_{topic.replace('*', 'wildcard').replace('.', '_')}",
            )
        except TypeError:
            # Compatibility with EventBus variants that use pattern=...
            subscription = subscribe(
                pattern=topic,
                handler=wrapped_handler,
                name=f"backtest_collector_{topic.replace('*', 'wildcard').replace('.', '_')}",
            )

        self._collector_subscriptions.append(subscription)

    def _wrap_collector_handler(self, topic: str, handler: Any) -> Any:
        async def _wrapped(event_or_payload: Any) -> None:
            payload = self._payload_from_event_or_dict(event_or_payload)
            event_topic = self._topic_from_event_or_fallback(event_or_payload, topic)

            if event_topic and "topic" not in payload:
                payload["topic"] = event_topic
            if event_topic and "event_topic" not in payload:
                payload["event_topic"] = event_topic

            result = handler(payload)

            if inspect.isawaitable(result):
                await result

        return _wrapped

    def _unregister_collectors(self) -> None:
        if not self._collector_subscriptions:
            return

        unsubscribe = getattr(self.components.event_bus, "unsubscribe", None)

        if not callable(unsubscribe):
            self._collector_subscriptions.clear()
            return

        for subscription in list(self._collector_subscriptions):
            try:
                unsubscribe(subscription)
            except Exception:
                self.logger.debug(
                    "Failed to unsubscribe backtest collector",
                    extra={"subscription": str(subscription)},
                )

        self._collector_subscriptions.clear()
        self._registered = False

    async def _collect_event_log(self, payload: dict[str, Any]) -> None:
        if not self.config.collect_event_log:
            return

        topic = str(payload.get("topic") or payload.get("event_topic") or "")
        timestamp = self._payload_timestamp(payload)

        self.collectors.events.append(
            BacktestEvent(
                event_id=str(payload.get("event_id") or payload.get("replay_event_id") or new_id("evt")),
                event_type=self._backtest_event_type_from_topic(topic),
                topic=topic,
                timestamp_ms=timestamp,
                payload=dict(payload),
                source=str(payload.get("source") or "event_bus"),
                sequence=int(payload.get("sequence") or payload.get("replay_sequence") or 0),
                is_warmup=bool(payload.get("is_warmup", False)),
                metadata={
                    "collected_by": "StrategyTester",
                    "topic": topic,
                },
            )
        )

    async def _collect_signal_generated(self, payload: dict[str, Any]) -> None:
        self.collectors.signals.append(
            self._build_signal_record(
                payload,
                outcome=SignalOutcome.GENERATED,
            )
        )

    async def _collect_signal_rejected(self, payload: dict[str, Any]) -> None:
        self.collectors.signals.append(
            self._build_signal_record(
                payload,
                outcome=SignalOutcome.REJECTED_BY_STRATEGY,
            )
        )

    async def _collect_signal_updated(self, payload: dict[str, Any]) -> None:
        self.collectors.signals.append(
            self._build_signal_record(
                payload,
                outcome=SignalOutcome.UNKNOWN,
            )
        )

    async def _collect_signal_confirmed(self, payload: dict[str, Any]) -> None:
        signal_id = self._value(payload, ["signal_id", "id"])
        self._update_signal_outcome(signal_id, SignalOutcome.CONFIRMED_BY_RISK)

        self.collectors.risk_decisions.append(
            BacktestRiskDecisionRecord(
                run_id=payload.get("run_id"),
                signal_id=signal_id,
                strategy_name=self._value(payload, ["strategy_name", "strategy"]),
                symbol=self._value(payload, ["symbol"]),
                timestamp_ms=self._payload_timestamp(payload),
                approved=True,
                blocked=False,
                reason=payload.get("reason"),
                risk_amount=self._float_value(payload, ["risk_amount", "final_risk_amount"]),
                final_size=self._float_value(payload, ["final_size", "quantity", "size"]),
                final_leverage=self._float_value(payload, ["final_leverage", "leverage"]),
                final_margin=self._float_value(payload, ["final_margin", "margin"]),
                final_notional=self._float_value(payload, ["final_notional", "notional"]),
                reservation_id=payload.get("reservation_id"),
                payload=payload,
                metadata={"source": "signal.confirmed"},
            )
        )

    async def _collect_risk_blocked(self, payload: dict[str, Any]) -> None:
        signal_id = self._value(payload, ["signal_id", "id"])
        self._update_signal_outcome(signal_id, SignalOutcome.BLOCKED_BY_RISK)

        self.collectors.risk_decisions.append(
            BacktestRiskDecisionRecord(
                run_id=payload.get("run_id"),
                signal_id=signal_id,
                strategy_name=self._value(payload, ["strategy_name", "strategy"]),
                symbol=self._value(payload, ["symbol"]),
                timestamp_ms=self._payload_timestamp(payload),
                approved=False,
                blocked=True,
                reason=payload.get("reason") or payload.get("block_reason"),
                payload=payload,
                metadata={"source": "risk.position_blocked"},
            )
        )

    async def _collect_risk_kill_switch(self, payload: dict[str, Any]) -> None:
        self.collectors.risk_decisions.append(
            BacktestRiskDecisionRecord(
                run_id=payload.get("run_id"),
                timestamp_ms=self._payload_timestamp(payload),
                approved=False,
                blocked=True,
                reason=payload.get("reason") or "kill_switch",
                payload=payload,
                metadata={"source": "risk.kill_switch"},
            )
        )

    async def _collect_risk_limit_warning(self, payload: dict[str, Any]) -> None:
        self.collectors.risk_decisions.append(
            BacktestRiskDecisionRecord(
                run_id=payload.get("run_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=self._value(payload, ["strategy_name", "strategy"]),
                symbol=self._value(payload, ["symbol"]),
                timestamp_ms=self._payload_timestamp(payload),
                approved=False,
                blocked=False,
                reason=payload.get("reason") or "limit_warning",
                payload=payload,
                metadata={"source": "risk.limit_warning", "warning": True},
            )
        )

    async def _collect_execution_event(self, payload: dict[str, Any]) -> None:
        topic = str(
            payload.get("topic")
            or payload.get("event_topic")
            or self._infer_execution_topic(payload)
        )

        self.collectors.execution_records.append(
            BacktestExecutionRecord(
                run_id=payload.get("run_id"),
                timestamp_ms=self._payload_timestamp(payload),
                topic=topic,
                order_id=payload.get("order_id"),
                fill_id=payload.get("fill_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                symbol=payload.get("symbol"),
                payload=payload,
                metadata={"source": "strategy_tester_collector"},
            )
        )

        signal_id = payload.get("signal_id")

        if topic.endswith("order_filled") or payload.get("status") == "filled":
            self._update_signal_outcome(signal_id, SignalOutcome.ORDER_FILLED)

    async def _collect_position_event(self, payload: dict[str, Any]) -> None:
        topic = str(
            payload.get("topic")
            or payload.get("event_topic")
            or self._infer_position_topic(payload)
        )

        self.collectors.position_records.append(
            BacktestPositionRecord(
                run_id=payload.get("run_id"),
                timestamp_ms=self._payload_timestamp(payload),
                topic=topic,
                position_id=payload.get("position_id"),
                signal_id=payload.get("signal_id"),
                strategy_name=payload.get("strategy_name"),
                symbol=payload.get("symbol"),
                payload=payload,
                metadata={"source": "strategy_tester_collector"},
            )
        )

        signal_id = payload.get("signal_id")

        if topic.endswith("position.opened") or payload.get("status") == "open":
            self._update_signal_outcome(signal_id, SignalOutcome.POSITION_OPENED)

        if topic.endswith("position.closed") or payload.get("status") == "closed":
            pnl = self._float_value(
                payload,
                ["net_realized_pnl", "realized_pnl", "pnl"],
                default=0.0,
            )

            if pnl > 0:
                outcome = SignalOutcome.POSITION_CLOSED_WIN
            elif pnl < 0:
                outcome = SignalOutcome.POSITION_CLOSED_LOSS
            else:
                outcome = SignalOutcome.POSITION_CLOSED_BREAKEVEN

            self._update_signal_outcome(signal_id, outcome, pnl=pnl)

    # -------------------------------------------------------------------------
    # Result construction
    # -------------------------------------------------------------------------

    def _create_result(self) -> BacktestResult:
        bt_config = self.backtest_config

        default_models = SimulationModelSnapshot()

        result = BacktestResult(
            run_id=new_id("bt"),
            run_name=self.config.run_name,
            mode=self.config.mode,
            status=BacktestStatus.CREATED,
            period=bt_config.period()
            if bt_config
            else (self.dataset.info.period if self.dataset else None),
            dataset_info=self.dataset.info if self.dataset else None,
            simulation_models=SimulationModelSnapshot(
                fill_model=bt_config.execution_simulator.fill_model
                if bt_config
                else default_models.fill_model,
                candle_execution_path=bt_config.execution_simulator.candle_execution_path
                if bt_config
                else default_models.candle_execution_path,
                slippage_model=bt_config.cost_model.slippage_model
                if bt_config
                else default_models.slippage_model,
                commission_model=bt_config.cost_model.commission_model
                if bt_config
                else default_models.commission_model,
                liquidity_model=bt_config.execution_simulator.liquidity_model
                if bt_config
                else default_models.liquidity_model,
                latency_model=bt_config.execution_simulator.latency_model
                if bt_config
                else default_models.latency_model,
                funding_mode=bt_config.cost_model.funding_mode
                if bt_config
                else default_models.funding_mode,
                position_accounting_mode=bt_config.position_simulator.position_accounting_mode
                if bt_config
                else default_models.position_accounting_mode,
                pnl_accounting_mode=bt_config.position_simulator.pnl_accounting_mode
                if bt_config
                else default_models.pnl_accounting_mode,
            ),
            initial_balance=bt_config.initial_balance if bt_config else 0.0,
            final_balance=bt_config.initial_balance if bt_config else 0.0,
            final_equity=bt_config.initial_balance if bt_config else 0.0,
            metadata={
                "config": bt_config.to_dict() if hasattr(bt_config, "to_dict") else None,
                "symbols": self.config.symbols,
                "timeframes": self.config.timeframes,
                "strategies": self.config.strategies,
                "test_all_registered_strategies": self.config.test_all_registered_strategies,
                "event_bus_type": self.components.event_bus.__class__.__name__,
            },
        )

        return result

    def _collect_results(self, result: BacktestResult) -> None:
        """
        Collect raw artifacts from simulators and collectors.
        """

        position_simulator = self.components.position_simulator
        execution_simulator = self.components.execution_simulator

        if execution_simulator is not None:
            result.orders = list(getattr(execution_simulator, "orders", {}).values())
            result.fills = list(getattr(execution_simulator, "fills", []))
            result.execution_records = list(getattr(execution_simulator, "records", []))

        if position_simulator is not None:
            result.positions = position_simulator.all_positions()
            result.trades = list(getattr(position_simulator, "trades", []))
            result.equity_curve = list(getattr(position_simulator, "equity_curve", []))
            result.position_records = list(getattr(position_simulator, "records", []))

            balance = getattr(position_simulator, "balance", None)
            if balance is not None:
                result.final_balance = getattr(balance, "cash_balance", result.final_balance)
                result.final_equity = getattr(balance, "equity", result.final_equity)

        if self.collectors.execution_records:
            known = {record.record_id for record in result.execution_records}
            result.execution_records.extend(
                record
                for record in self.collectors.execution_records
                if record.record_id not in known
            )

        if self.collectors.position_records:
            known = {record.record_id for record in result.position_records}
            result.position_records.extend(
                record
                for record in self.collectors.position_records
                if record.record_id not in known
            )

        result.signals = list(self.collectors.signals)
        result.risk_decisions = list(self.collectors.risk_decisions)

        if self.config.collect_event_log:
            result.events = list(self.collectors.events)

    def _calculate_metrics(self, result: BacktestResult) -> None:
        metrics = self.components.performance_metrics

        if metrics is None:
            raise BacktestResultCollectionError("PerformanceMetrics component is missing.")

        metrics_input = build_metrics_input_from_components(
            initial_balance=result.initial_balance,
            final_balance=result.final_balance,
            final_equity=result.final_equity,
            trades=result.trades,
            positions=result.positions,
            equity_curve=result.equity_curve,
            signals=result.signals,
            risk_decisions=result.risk_decisions,
            orders=result.orders,
            fills=result.fills,
            execution_records=result.execution_records,
            metadata={
                "run_id": result.run_id,
                "run_name": result.run_name,
            },
        )

        result.portfolio = metrics.calculate_portfolio_result(metrics_input)

    def _calculate_model_analytics(self, result: BacktestResult) -> None:
        engine = self.components.model_analytics

        if engine is None:
            return

        result.analytics = engine.analyze(
            ModelAnalyticsInput(
                signals=result.signals,
                risk_decisions=result.risk_decisions,
                orders=result.orders,
                fills=result.fills,
                positions=result.positions,
                trades=result.trades,
                execution_records=result.execution_records,
                metadata={
                    "run_id": result.run_id,
                    "run_name": result.run_name,
                },
            )
        )

    def _build_report(self, result: BacktestResult) -> None:
        builder = self.components.report_builder

        if builder is None:
            return

        if self.backtest_config is not None and not self.backtest_config.save_report:
            return

        builder.build(result)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_signal_record(
        self,
        payload: dict[str, Any],
        *,
        outcome: SignalOutcome,
    ) -> BacktestSignalRecord:
        return BacktestSignalRecord(
            run_id=payload.get("run_id"),
            signal_id=self._value(payload, ["signal_id", "id"]),
            strategy_name=self._value(payload, ["strategy_name", "strategy"]),
            symbol=self._value(payload, ["symbol"]),
            timeframe=self._value(payload, ["timeframe"]),
            side=self._value(payload, ["side", "signal_side", "direction"]),
            setup_type=self._value(payload, ["setup_type", "setup"]),
            confidence=self._float_value(payload, ["confidence"]),
            strength=self._float_value(payload, ["strength", "score"]),
            generated_at_ms=self._payload_timestamp(payload),
            outcome=outcome,
            payload=payload,
            metadata={"source": "strategy_tester_collector"},
        )

    def _update_signal_outcome(
        self,
        signal_id: str | None,
        outcome: SignalOutcome,
        *,
        pnl: float | None = None,
    ) -> None:
        if not signal_id:
            return

        for signal in reversed(self.collectors.signals):
            if signal.signal_id != signal_id:
                continue

            signal.outcome = outcome

            if outcome == SignalOutcome.CONFIRMED_BY_RISK:
                signal.confirmed_at_ms = self._now_ms()
            elif outcome == SignalOutcome.BLOCKED_BY_RISK:
                signal.rejected_at_ms = self._now_ms()
            elif outcome == SignalOutcome.POSITION_OPENED:
                signal.opened_at_ms = self._now_ms()
            elif outcome in {
                SignalOutcome.POSITION_CLOSED_WIN,
                SignalOutcome.POSITION_CLOSED_LOSS,
                SignalOutcome.POSITION_CLOSED_BREAKEVEN,
            }:
                signal.closed_at_ms = self._now_ms()

            if pnl is not None:
                signal.pnl = pnl

            return

    def _resolve_final_balance(self) -> float:
        position_simulator = self.components.position_simulator

        if position_simulator is not None:
            balance = getattr(position_simulator, "balance", None)
            if balance is not None:
                return float(getattr(balance, "cash_balance", 0.0))

        if self.result is not None:
            return self.result.final_balance

        return 0.0

    def _resolve_final_equity(self) -> float:
        position_simulator = self.components.position_simulator

        if position_simulator is not None:
            balance = getattr(position_simulator, "balance", None)
            if balance is not None:
                return float(getattr(balance, "equity", 0.0))

        if self.result is not None:
            return self.result.final_equity

        return 0.0

    def _now_ms(self) -> int:
        clock = self.components.clock

        if clock is not None and getattr(clock, "started", False):
            return clock.timestamp_ms()

        return timestamp_ms(utcnow())

    def _payload_timestamp(self, payload: dict[str, Any]) -> int:
        for key in (
            "timestamp_ms",
            "generated_at_ms",
            "created_at_ms",
            "updated_at_ms",
            "submitted_at_ms",
            "filled_at_ms",
            "closed_at_ms",
        ):
            value = payload.get(key)
            if value is not None:
                try:
                    return int(float(value))
                except Exception:
                    continue

        return self._now_ms()

    @staticmethod
    def _payload_from_event_or_dict(event_or_payload: Any) -> dict[str, Any]:
        if isinstance(event_or_payload, dict):
            return dict(event_or_payload)

        payload = getattr(event_or_payload, "payload", None)
        if isinstance(payload, dict):
            return dict(payload)

        return {}

    @staticmethod
    def _topic_from_event_or_fallback(event_or_payload: Any, fallback: str) -> str:
        for attr in ("topic", "name", "event_name", "type"):
            value = getattr(event_or_payload, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return fallback

    @staticmethod
    def _value(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value

        for nested_key in (
            "signal",
            "payload",
            "risk_decision",
            "execution_intent",
            "order",
            "fill",
            "position",
            "metadata",
        ):
            nested = payload.get(nested_key)
            if not isinstance(nested, dict):
                continue

            for key in keys:
                value = nested.get(key)
                if value is not None:
                    return value

        return default

    @classmethod
    def _float_value(
        cls,
        payload: dict[str, Any],
        keys: list[str],
        default: float | None = None,
    ) -> float | None:
        value = cls._value(payload, keys, default=default)

        if value is None:
            return default

        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _infer_execution_topic(payload: dict[str, Any]) -> str:
        status = str(payload.get("status") or "").lower()

        if status == "rejected":
            return "execution.order_rejected"
        if status == "cancelled":
            return "execution.order_cancelled"
        if status == "partially_filled":
            return "execution.order_partially_filled"
        if status == "filled":
            return "execution.order_filled"
        if status == "failed":
            return "execution.order_failed"

        return "execution.order_submitted"

    @staticmethod
    def _infer_position_topic(payload: dict[str, Any]) -> str:
        status = str(payload.get("status") or "").lower()

        if status == "open":
            return "position.opened"
        if status == "closed":
            return "position.closed"
        if status == "liquidated":
            return "position.liquidated"

        return "position.updated"

    @staticmethod
    def _component_has_attr(component: Any, names: list[str]) -> bool:
        if component is None:
            return False

        return any(getattr(component, name, None) is not None for name in names)

    @staticmethod
    def _get_registered_strategies(registry: Any) -> list[Any]:
        for method_name in ("all", "all_strategies", "strategies", "list_strategies"):
            method = getattr(registry, method_name, None)
            if not callable(method):
                continue

            result = method()

            if isinstance(result, dict):
                return list(result.values())

            if isinstance(result, list):
                return result

            if isinstance(result, tuple):
                return list(result)

        raw = getattr(registry, "_strategies", None)
        if isinstance(raw, dict):
            return list(raw.values())

        return []

    @staticmethod
    def _strategy_name(strategy: Any) -> str:
        for attr in ("strategy_name", "name"):
            value = getattr(strategy, attr, None)
            if isinstance(value, str) and value:
                return value

        return strategy.__class__.__name__

    @staticmethod
    def _backtest_event_type_from_topic(topic: str) -> Any:
        # Import here to avoid broad enum dependency at module import time.
        from backtesting.enums import BacktestEventType

        normalized = topic.lower()

        if normalized.startswith("market."):
            return BacktestEventType.MARKET
        if normalized.startswith("analytics."):
            return BacktestEventType.ANALYTICS
        if normalized.startswith("strategy."):
            return BacktestEventType.STRATEGY
        if normalized.startswith("signal."):
            return BacktestEventType.SIGNAL
        if normalized.startswith("risk."):
            return BacktestEventType.RISK
        if normalized.startswith("execution."):
            return BacktestEventType.EXECUTION
        if normalized.startswith("position."):
            return BacktestEventType.POSITION

        return BacktestEventType.SYSTEM

    def stats(self) -> dict[str, Any]:
        return {
            "prepared": self._prepared,
            "registered": self._registered,
            "running": self._running,
            "event_bus_type": self.components.event_bus.__class__.__name__,
            "collector_subscriptions": len(self._collector_subscriptions),
            "dataset_events": len(self.dataset.events) if self.dataset else 0,
            "data_caches": [item.__class__.__name__ for item in self.components.data_caches],
            "analytics_components": [
                item.__class__.__name__ for item in self.components.analytics_components
            ],
            "has_strategy_engine": self.components.strategy_engine is not None,
            "has_signal_processor": self.components.signal_processor is not None,
            "has_risk_manager": self.components.risk_manager is not None,
            "has_market_replay": self.components.market_replay is not None,
            "has_execution_simulator": self.components.execution_simulator is not None,
            "has_position_simulator": self.components.position_simulator is not None,
            "collected": {
                "events": len(self.collectors.events),
                "signals": len(self.collectors.signals),
                "risk_decisions": len(self.collectors.risk_decisions),
                "execution_records": len(self.collectors.execution_records),
                "position_records": len(self.collectors.position_records),
            },
        }


__all__ = [
    "BacktestComponentBundle",
    "BacktestCollectors",
    "StrategyTester",
]