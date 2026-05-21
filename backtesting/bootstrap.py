"""
Backtesting project bootstrap.

This module builds the production-style trading pipeline for historical
backtests. It does not replay data, does not call strategies directly, does not
approve risk decisions and does not simulate execution. It only constructs and
wires the same event-driven project components that are used by the live system,
so StrategyTester can run:

    MarketReplay -> EventBus market.*
        -> data caches -> market.*.updated
        -> analytics -> analytics.*
        -> StrategyEngine / SignalProcessor -> signal.*
        -> RiskManager -> signal.confirmed / risk.*
        -> ExecutionSimulator -> execution.*
        -> PositionSimulator -> position.*

Design rules:
- no live exchange adapters;
- no live TradeExecutor / OrderManager / PositionManager;
- no dynamic broad auto-discovery of all analytics modules;
- analytics bootstrap is explicit and domain-scoped;
- components communicate through EventBus only.
"""

from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from backtesting.config import BacktestConfig
from backtesting.enums import BacktestDataType
from backtesting.exceptions import BacktestDependencyError


ComponentFactory = Callable[..., Any]


# =============================================================================
# Bootstrap DTOs
# =============================================================================


@dataclass(slots=True)
class BootstrapFailure:
    """
    One non-fatal bootstrap failure.

    Failures are collected first and only raised by the domain-level validation
    step when the missing component is required for the current backtest.
    """

    component: str
    reason: str

    def format(self) -> str:
        return f"- {self.component}: {self.reason}"


@dataclass(slots=True)
class BootstrapDiagnostics:
    """
    Human-readable bootstrap diagnostics.
    """

    loaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[BootstrapFailure] = field(default_factory=list)

    def add_loaded(self, category: str, component: Any) -> None:
        self.loaded.append(f"[{category}] {qualified_name(component.__class__)}")

    def add_skipped(self, component: str, reason: str) -> None:
        self.skipped.append(f"- {component}: {reason}")

    def add_failure(self, component: str, reason: str) -> None:
        self.failed.append(BootstrapFailure(component=component, reason=reason))

    def format(self) -> str:
        lines: list[str] = ["", "========== BACKTEST BOOTSTRAP =========="]

        lines.append("Loaded:")
        if self.loaded:
            lines.extend(f"- {item}" for item in self.loaded)
        else:
            lines.append("- <none>")

        if self.skipped:
            lines.append("Skipped:")
            lines.extend(self.skipped)

        if self.failed:
            lines.append("Failed:")
            lines.extend(item.format() for item in self.failed)

        lines.append("========================================")
        return "\n".join(lines)


@dataclass(slots=True)
class BuiltStrategyPipeline:
    """
    Built strategy-layer components.
    """

    registry: Any
    runtime_state: Any
    signal_processor: Any
    strategy_engine: Any


@dataclass(slots=True)
class BuiltBacktestPipeline:
    """
    Complete production-style pipeline needed by StrategyTester.
    """

    core_config: Config
    event_bus: EventBus
    scheduler: Scheduler

    data_caches: list[Any]
    analytics_components: list[Any]

    strategy_pipeline: BuiltStrategyPipeline
    risk_manager: Any

    diagnostics: BootstrapDiagnostics = field(default_factory=BootstrapDiagnostics)

    @property
    def strategy_registry(self) -> Any:
        return self.strategy_pipeline.registry

    @property
    def signal_processor(self) -> Any:
        return self.strategy_pipeline.signal_processor

    @property
    def strategy_engine(self) -> Any:
        return self.strategy_pipeline.strategy_engine


@dataclass(slots=True)
class BacktestProjectBootstrapConfig:
    """
    Runtime bootstrap config.

    This config describes which production domains should be wired for the
    backtest. It is intentionally separate from BacktestConfig and production
    StrategyConfig/RiskConfig.
    """

    exchange: str = "binance"
    market_type: str = "usdm_futures"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1m"])

    enable_candles: bool = True
    enable_trades: bool = False
    enable_orderbook: bool = False
    enable_funding: bool = True
    enable_open_interest: bool = True
    enable_liquidations: bool = False
    enable_spreads: bool = False

    require_analytics_for_enabled_streams: bool = True
    verbose_bootstrap_errors: bool = False

    service_name: str = "backtesting.bootstrap"

    @classmethod
    def from_env(cls) -> BacktestProjectBootstrapConfig:
        symbols = env_list("BACKTEST_SYMBOLS", ["BTCUSDT", "DOGEUSDT", "SOLUSDT"])
        timeframes = env_list("BACKTEST_TIMEFRAMES", ["1m"])

        return cls(
            exchange=os.getenv("BACKTEST_EXCHANGE", "binance").strip().lower(),
            market_type=os.getenv("BACKTEST_MARKET_TYPE", "usdm_futures").strip().lower(),
            symbols=symbols,
            timeframes=timeframes,
            enable_candles=env_bool("BACKTEST_USE_CANDLES", True),
            enable_trades=env_bool("BACKTEST_USE_TRADES", False),
            enable_orderbook=env_bool("BACKTEST_USE_ORDERBOOK", False),
            enable_funding=env_bool("BACKTEST_USE_FUNDING", True),
            enable_open_interest=env_bool("BACKTEST_USE_OPEN_INTEREST", True),
            enable_liquidations=env_bool("BACKTEST_USE_LIQUIDATIONS", False),
            enable_spreads=env_bool("BACKTEST_USE_SPREADS", False),
            require_analytics_for_enabled_streams=env_bool(
                "BACKTEST_REQUIRE_ANALYTICS",
                True,
            ),
            verbose_bootstrap_errors=env_bool(
                "BACKTEST_VERBOSE_BOOTSTRAP_ERRORS",
                False,
            ),
        )

    @property
    def first_symbol(self) -> str:
        return self.symbols[0] if self.symbols else "BTCUSDT"

    @property
    def first_timeframe(self) -> str:
        return self.timeframes[0] if self.timeframes else "1m"

    @property
    def enabled_data_types(self) -> set[BacktestDataType]:
        result: set[BacktestDataType] = set()
        if self.enable_candles:
            result.add(BacktestDataType.CANDLES)
        if self.enable_trades:
            result.add(BacktestDataType.TRADES)
        if self.enable_orderbook:
            result.add(BacktestDataType.ORDERBOOK)
            result.add(BacktestDataType.ORDERBOOK_SNAPSHOT)
        if self.enable_funding:
            result.add(BacktestDataType.FUNDING)
        if self.enable_open_interest:
            result.add(BacktestDataType.OPEN_INTEREST)
        if self.enable_liquidations:
            result.add(BacktestDataType.LIQUIDATIONS)
        return result

    def validate(self) -> None:
        self.exchange = self.exchange.strip().lower()
        self.market_type = self.market_type.strip().lower()
        self.symbols = unique_strings(symbol.upper() for symbol in self.symbols if symbol.strip())
        self.timeframes = unique_strings(tf.strip() for tf in self.timeframes if tf.strip())

        if not self.symbols:
            raise BacktestDependencyError("At least one symbol is required for backtest bootstrap.")

        if not self.timeframes:
            raise BacktestDependencyError("At least one timeframe is required for backtest bootstrap.")


# =============================================================================
# Main bootstrap
# =============================================================================


class BacktestProjectBootstrap:
    """
    Builds production project components for a historical backtest.

    The bootstrap is intentionally not a runner. It returns BuiltBacktestPipeline
    and lets StrategyTester handle lifecycle, MarketReplay and result collection.
    """

    def __init__(
        self,
        config: BacktestProjectBootstrapConfig | None = None,
        *,
        core_config: Config | None = None,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        backtest_config: BacktestConfig | None = None,
        analytics_factories: Sequence[ComponentFactory] | None = None,
        logger_name: str = "backtesting.bootstrap",
    ) -> None:
        self.config = config or BacktestProjectBootstrapConfig()
        self.config.validate()

        self.core_config = core_config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.backtest_config = backtest_config

        self.analytics_factories = list(analytics_factories or [])
        self.logger = get_logger(logger_name)

        self.diagnostics = BootstrapDiagnostics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> BuiltBacktestPipeline:
        """
        Build all components needed by StrategyTester.
        """

        core_config = self.core_config or self.build_core_config()
        event_bus = self.require_system_event_bus()
        scheduler = self.require_system_scheduler()

        # Keep shared system runtime on the bootstrap instance so every later
        # helper uses the same EventBus/Scheduler. Full-pipeline backtests must
        # never silently create a local bus/scheduler.
        self.core_config = core_config
        self.event_bus = event_bus
        self.scheduler = scheduler

        data_caches = self.build_data_caches(
            core_config=core_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        analytics_components = self.build_analytics_components(
            core_config=core_config,
            event_bus=event_bus,
            scheduler=scheduler,
            data_caches=data_caches,
        )

        strategy_pipeline = self.build_strategy_pipeline(
            core_config=core_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        risk_manager = self.build_risk_manager(
            core_config=core_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        self.validate_pipeline(
            data_caches=data_caches,
            analytics_components=analytics_components,
            strategy_pipeline=strategy_pipeline,
            risk_manager=risk_manager,
        )

        return BuiltBacktestPipeline(
            core_config=core_config,
            event_bus=event_bus,
            scheduler=scheduler,
            data_caches=data_caches,
            analytics_components=analytics_components,
            strategy_pipeline=strategy_pipeline,
            risk_manager=risk_manager,
            diagnostics=self.diagnostics,
        )

    # ------------------------------------------------------------------
    # Core runtime
    # ------------------------------------------------------------------

    @staticmethod
    def build_core_config() -> Config:
        return Config.from_env()

    def require_system_event_bus(self) -> EventBus:
        if self.event_bus is None:
            raise BacktestDependencyError(
                "BacktestProjectBootstrap requires the shared system EventBus. "
                "Pass event_bus=... from run.py / application runtime; bootstrap "
                "must not create a local EventBus for full-pipeline backtests."
            )
        return self.event_bus

    def require_system_scheduler(self) -> Scheduler:
        if self.scheduler is None:
            raise BacktestDependencyError(
                "BacktestProjectBootstrap requires the shared system Scheduler. "
                "Pass scheduler=... from run.py / application runtime; bootstrap "
                "must not create a local Scheduler for full-pipeline backtests."
            )
        return self.scheduler

    @staticmethod
    def build_event_bus(core_config: Config) -> EventBus:
        raise BacktestDependencyError(
            "BacktestProjectBootstrap.build_event_bus() is disabled. "
            "Use the shared system EventBus and pass event_bus=... explicitly."
        )

    @staticmethod
    def build_scheduler(event_bus: EventBus) -> Scheduler:
        raise BacktestDependencyError(
            "BacktestProjectBootstrap.build_scheduler() is disabled. "
            "Use the shared system Scheduler and pass scheduler=... explicitly."
        )

    # ------------------------------------------------------------------
    # Data layer
    # ------------------------------------------------------------------

    def build_data_caches(
        self,
        *,
        core_config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> list[Any]:
        """
        Build production data caches.

        These components must listen to raw market.* events and emit cached
        market.*.updated events. They are not backtesting-specific.
        """

        components: list[Any] = []

        if self.config.enable_candles:
            components.append(
                self.instantiate_required(
                    "data.candles_cache.CandlesCache",
                    {
                        "config": core_config,
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                    },
                )
            )

        if self.config.enable_trades:
            components.append(
                self.instantiate_required(
                    "data.trades_cache.TradesCache",
                    {
                        "config": core_config,
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                    },
                )
            )

        if self.config.enable_orderbook:
            components.append(
                self.instantiate_required(
                    "data.orderbook_cache.OrderBookCache",
                    {
                        "config": core_config,
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                    },
                )
            )

        if self.config.enable_funding:
            components.append(
                self.instantiate_required(
                    "data.funding_cache.FundingCache",
                    {
                        "config": core_config,
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                    },
                )
            )

        if self.config.enable_open_interest:
            components.append(
                self.instantiate_required(
                    "data.open_interest_cache.OpenInterestCache",
                    {
                        "config": core_config,
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                    },
                )
            )

        return dedupe_components(components)

    # ------------------------------------------------------------------
    # Analytics layer
    # ------------------------------------------------------------------

    def build_analytics_components(
        self,
        *,
        core_config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        data_caches: Sequence[Any],
    ) -> list[Any]:
        """
        Build explicit analytics components for enabled streams.

        There is no broad analytics auto-discovery here. The candidates are
        deliberate domain entry points. This prevents unrelated domains from
        being instantiated when the dataset does not contain their input stream.
        """

        components: list[Any] = []

        available = self.analytics_available_kwargs(
            core_config=core_config,
            event_bus=event_bus,
            scheduler=scheduler,
            data_caches=data_caches,
        )

        for factory in self.analytics_factories:
            component_name = qualified_name(factory)
            try:
                component = factory(**available)
            except (TypeError, ValueError, AttributeError) as exc:
                self.diagnostics.add_failure(component_name, f"factory failed: {exc}")
                continue

            components.append(component)
            self.diagnostics.add_loaded("custom", component)

        if self.config.enable_candles:
            components.extend(self.build_price_action_analytics(available))

        if self.config.enable_funding:
            components.extend(self.build_funding_analytics(available))

        if self.config.enable_open_interest:
            components.extend(self.build_open_interest_analytics(available))

        if self.config.enable_liquidations:
            components.extend(self.build_liquidations_analytics(available))

        if self.config.enable_trades:
            components.extend(self.build_orderflow_analytics(available))
            components.extend(self.build_whales_analytics(available))

        if self.config.enable_orderbook:
            components.extend(self.build_liquidity_analytics(available))
            components.extend(self.build_spoofing_analytics(available))

        if self.config.enable_spreads:
            components.extend(self.build_spreads_analytics(available))

        components = dedupe_components(components)
        self.validate_required_analytics(components)
        return components

    def analytics_available_kwargs(
        self,
        *,
        core_config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        data_caches: Sequence[Any],
    ) -> dict[str, Any]:
        candles_cache = find_component(data_caches, "CandlesCache")
        trades_cache = find_component(data_caches, "TradesCache")
        orderbook_cache = find_component(data_caches, "OrderBookCache")
        funding_cache = find_component(data_caches, "FundingCache")
        open_interest_cache = find_component(data_caches, "OpenInterestCache")

        return {
            # IMPORTANT:
            # Do not pass core.config.Config as plain "config" into analytics.
            # Analytics constructors use "config" for their own domain config
            # types, e.g. FundingAnalyzerConfig, OIAnalyzerConfig or
            # PriceActionAnalyzerConfig. Passing the global core Config there
            # causes runtime AttributeError such as missing service_name,
            # max_candles, enable_market_structure or
            # assert_production_topics_allowed.
            #
            # Keep the global app config available under explicit names only.
            "config": None,
            "core_config": core_config,
            "app_config": core_config,
            "event_bus": event_bus,
            "scheduler": scheduler,
            "data_caches": list(data_caches),
            "caches": list(data_caches),
            "candles_cache": candles_cache,
            "trades_cache": trades_cache,
            "orderbook_cache": orderbook_cache,
            "funding_cache": funding_cache,
            "open_interest_cache": open_interest_cache,
            "exchange": self.config.exchange,
            "default_exchange": self.config.exchange,
            "market_type": self.config.market_type,
            "default_market_type": self.config.market_type,
            "symbols": list(self.config.symbols),
            "symbol": self.config.first_symbol,
            "default_symbol": self.config.first_symbol,
            "timeframes": list(self.config.timeframes),
            "timeframe": self.config.first_timeframe,
            "exchange_symbol": self.config.first_symbol,
            "service_name": self.config.service_name,
        }

    def build_price_action_analytics(self, available: dict[str, Any]) -> list[Any]:
        """
        Build candle-driven price action analytics.

        Prefer the domain facade if present. If a facade is unavailable, try
        concrete child analyzers that listen to candle/cache events themselves.
        """

        return self.instantiate_optional_candidates(
            category="price_action",
            candidates=[
                "analytics.price_action.price_action_analyzer.PriceActionAnalyzer",
                "analytics.price_action.market_structure.MarketStructureAnalyzer",
                "analytics.price_action.trend.TrendAnalyzer",
                "analytics.price_action.fair_value_gap.FairValueGapAnalyzer",
                "analytics.price_action.support_resistance.SupportResistanceAnalyzer",
                "analytics.price_action.liquidity_levels.LiquidityLevelsAnalyzer",
            ],
            available=available,
            require_cache="CandlesCache",
        )

    def build_funding_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="funding",
            candidates=[
                "analytics.funding.funding_analyzer.FundingAnalyzer",
            ],
            available=available,
            require_cache="FundingCache",
        )

    def build_open_interest_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="open_interest",
            candidates=[
                "analytics.open_interest.oi_analyzer.OIAnalyzer",
            ],
            available=available,
            require_cache="OpenInterestCache",
        )

    def build_liquidations_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="liquidations",
            candidates=[
                "analytics.liquidations.liquidation_analyzer.LiquidationAnalyzer",
                "analytics.liquidations.analyzer.LiquidationAnalyzer",
            ],
            available=available,
        )

    def build_orderflow_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="orderflow",
            candidates=[
                "analytics.orderflow.orderflow_analyzer.OrderFlowAnalyzer",
                "analytics.orderflow.cvd_analyzer.CVDAnalyzer",
            ],
            available=available,
            require_cache="TradesCache",
        )

    def build_liquidity_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="liquidity",
            candidates=[
                "analytics.liquidity.liquidity_analyzer.LiquidityAnalyzer",
                "analytics.liquidity.liquidity_levels.LiquidityLevelsAnalyzer",
            ],
            available=available,
            require_cache="OrderBookCache",
        )

    def build_spoofing_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="spoofing",
            candidates=[
                "analytics.spoofing.spoofing_analyzer.SpoofingAnalyzer",
            ],
            available=available,
            require_cache="OrderBookCache",
        )

    def build_whales_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="whales",
            candidates=[
                "analytics.whales.large_trade_detector.LargeTradeDetector",
                "analytics.whales.whale_activity_analyzer.WhaleActivityAnalyzer",
            ],
            available=available,
            require_cache="TradesCache",
        )

    def build_spreads_analytics(self, available: dict[str, Any]) -> list[Any]:
        return self.instantiate_optional_candidates(
            category="spreads",
            candidates=[
                "analytics.spreads.spread_analyzer.SpreadAnalyzer",
                "analytics.spreads.cross_exchange_analyzer.CrossExchangeSpreadAnalyzer",
                "analytics.spreads.spot_futures_analyzer.SpotFuturesSpreadAnalyzer",
            ],
            available=available,
        )

    def instantiate_optional_candidates(
        self,
        *,
        category: str,
        candidates: Sequence[str],
        available: dict[str, Any],
        require_cache: str | None = None,
    ) -> list[Any]:
        if require_cache is not None and find_named_value(available, require_cache) is None:
            for path in candidates:
                self.diagnostics.add_skipped(path, f"requires {require_cache}, but cache is not enabled")
            return []

        components: list[Any] = []

        for path in candidates:
            component = self.instantiate_optional(path, available)
            if component is None:
                continue

            components.append(component)
            self.diagnostics.add_loaded(category, component)

            # The facade components are enough for these domains. Avoid
            # instantiating every child analyzer twice.
            if path.endswith("PriceActionAnalyzer") or path.endswith("FundingAnalyzer") or path.endswith("OIAnalyzer"):
                break

        return components

    def validate_required_analytics(self, components: Sequence[Any]) -> None:
        if not self.config.require_analytics_for_enabled_streams:
            return

        loaded_names = {
            qualified_name(component.__class__).lower()
            for component in components
        }

        missing: list[str] = []

        if self.config.enable_candles and not any("price_action" in name for name in loaded_names):
            missing.append("price_action")

        if self.config.enable_funding and not any("funding" in name for name in loaded_names):
            missing.append("funding")

        if self.config.enable_open_interest and not any(
            "open_interest" in name or "oi_analyzer" in name for name in loaded_names
        ):
            missing.append("open_interest")

        if self.config.enable_trades and not any(
            "orderflow" in name or "whales" in name for name in loaded_names
        ):
            missing.append("orderflow/trades")

        if self.config.enable_orderbook and not any(
            "liquidity" in name or "spoofing" in name for name in loaded_names
        ):
            missing.append("liquidity/orderbook")

        if self.config.enable_liquidations and not any("liquidation" in name for name in loaded_names):
            missing.append("liquidations")

        if not missing:
            return

        diagnostics = self.diagnostics.format()
        missing_text = ", ".join(missing)
        raise BacktestDependencyError(
            "Backtest bootstrap could not build required analytics "
            f"for enabled streams: {missing_text}.\n{diagnostics}"
        )

    # ------------------------------------------------------------------
    # Strategy layer
    # ------------------------------------------------------------------

    def build_strategy_pipeline(
        self,
        *,
        core_config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> BuiltStrategyPipeline:
        """
        Build production strategy pipeline.

        StrategyEngine and SignalProcessor are responsible for listening to
        analytics.* events and publishing signal.* events. Bootstrap does not
        call strategy.generate_signal directly.
        """

        strategy_config = self.build_strategy_config()
        registry = self.build_strategy_registry(
            strategy_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        self.validate_strategy_registry(registry)
        runtime_state = self.build_strategy_runtime_state(strategy_config)
        signal_processor = self.build_signal_processor(
            strategy_config=strategy_config,
            registry=registry,
            runtime_state=runtime_state,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        strategy_engine = self.build_strategy_engine(
            strategy_config=strategy_config,
            registry=registry,
            runtime_state=runtime_state,
            signal_processor=signal_processor,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        return BuiltStrategyPipeline(
            registry=registry,
            runtime_state=runtime_state,
            signal_processor=signal_processor,
            strategy_engine=strategy_engine,
        )

    def build_strategy_config(self) -> Any:
        """
        Build StrategyConfig for full-pipeline backtesting.

        Important:
        StrategyEventHandler subscribes only to keys from
        config.routing.event_to_categories when that mapping is not empty.
        Therefore the routing map must contain the real analytics topics emitted
        by the current analytics package, otherwise analytics.* events will be
        produced but StrategyEngine will never receive them.
        """

        candidates = [
            "strategy.presets.build_default_strategy_config",
            "strategy.config.StrategyConfig",
        ]

        last_error: str | None = None
        for path in candidates:
            target = import_object_or_none(path)
            if target is None:
                last_error = f"{path} is not importable"
                continue

            try:
                if inspect.isclass(target):
                    config = target()
                else:
                    preset_name = os.getenv("BACKTEST_STRATEGY_PRESET", "intraday").strip() or "intraday"
                    try:
                        config = target(
                            symbols=list(self.config.symbols),
                            preset_name=preset_name,
                            use_required_features=env_bool(
                                "BACKTEST_STRATEGY_USE_REQUIRED_FEATURES",
                                False,
                            ),
                        )
                    except TypeError:
                        config = target()

                self.patch_strategy_event_routing(config)
                return config

            except (TypeError, ValueError, AttributeError) as exc:
                last_error = f"{path} failed: {exc}"

        raise BacktestDependencyError(
            "Could not build StrategyConfig.",
            details={"last_error": last_error},
        )

    def patch_strategy_event_routing(self, strategy_config: Any) -> None:
        """
        Make StrategyEventHandler subscribe to the actual analytics topics.

        Current analytics emits topics such as:
        - analytics.price_action.updated
        - analytics.price_action.trend.updated
        - analytics.price_action.market_structure.bos
        - analytics.oi.updated
        - analytics.oi.divergence

        If these keys are missing from routing.event_to_categories, the
        StrategyEventHandler will not subscribe to them because it uses the
        configured mapping keys as subscription topics.
        """

        routing = getattr(strategy_config, "routing", None)
        if routing is None:
            return

        event_to_categories = getattr(routing, "event_to_categories", None)
        if not isinstance(event_to_categories, dict):
            try:
                routing.event_to_categories = {}
                event_to_categories = routing.event_to_categories
            except (AttributeError, TypeError):
                return

        try:
            strategy_category = import_required("strategy.enums.StrategyCategory")
            price_action = strategy_category.PRICE_ACTION
            funding = strategy_category.FUNDING
            open_interest = strategy_category.OPEN_INTEREST
            orderflow = getattr(strategy_category, "ORDERFLOW", price_action)
            liquidity = getattr(strategy_category, "LIQUIDITY", price_action)
            liquidations = getattr(strategy_category, "LIQUIDATIONS", price_action)
            spreads = getattr(strategy_category, "SPREADS", price_action)
        except (ImportError, AttributeError, BacktestDependencyError):
            price_action = "price_action"
            funding = "funding"
            open_interest = "open_interest"
            orderflow = "orderflow"
            liquidity = "liquidity"
            liquidations = "liquidations"
            spreads = "spreads"

        topic_map: dict[str, list[Any]] = {
            # Price action facade and child analyzers.
            "analytics.price_action.updated": [price_action],
            "analytics.price_action.trend.updated": [price_action],
            "analytics.price_action.trend.trend_alignment": [price_action],
            "analytics.price_action.trend.trend_started": [price_action],
            "analytics.price_action.trend.trend_reversal": [price_action],
            "analytics.price_action.market_structure.updated": [price_action],
            "analytics.price_action.market_structure.bos": [price_action],
            "analytics.price_action.market_structure.choch": [price_action],
            "analytics.price_action.market_structure.hh": [price_action],
            "analytics.price_action.market_structure.hl": [price_action],
            "analytics.price_action.market_structure.lh": [price_action],
            "analytics.price_action.market_structure.ll": [price_action],
            "analytics.price_action.market_structure.swing_high": [price_action],
            "analytics.price_action.market_structure.swing_low": [price_action],
            "analytics.price_action.fair_value_gap.updated": [price_action],
            "analytics.price_action.fair_value_gap.fvg_created": [price_action],
            "analytics.price_action.fair_value_gap.fvg_fill_started": [price_action],
            "analytics.price_action.fair_value_gap.fvg_partially_filled": [price_action],
            "analytics.price_action.fair_value_gap.fvg_filled": [price_action],
            "analytics.price_action.fair_value_gap.fvg_respected": [price_action],
            "analytics.price_action.fair_value_gap.fvg_retested": [price_action],
            "analytics.price_action.fair_value_gap.fvg_merged": [price_action],
            "analytics.price_action.support_resistance.updated": [price_action],
            "analytics.price_action.support_resistance.level_created": [price_action],
            "analytics.price_action.support_resistance.level_merged": [price_action],
            "analytics.price_action.liquidity_levels.updated": [price_action],
            "analytics.price_action.liquidity_levels.level_created": [liquidity, price_action],
            "analytics.price_action.liquidity_levels.level_merged": [liquidity, price_action],

            # Funding analytics.
            "analytics.funding.updated": [funding],
            "analytics.funding.snapshot": [funding],
            "analytics.funding.signal": [funding],
            "analytics.funding.regime": [funding],
            "analytics.funding.flip": [funding],
            "analytics.funding.extreme": [funding],
            "analytics.funding.divergence": [funding],

            # Open interest analytics aliases used by the current package.
            "analytics.oi.updated": [open_interest],
            "analytics.oi.anomaly": [open_interest],
            "analytics.oi.capitulation": [open_interest],
            "analytics.oi.divergence": [open_interest],
            "analytics.oi.regime_changed": [open_interest],
            "analytics.oi.squeeze_setup": [open_interest],

            # Canonical open-interest aliases too.
            "analytics.open_interest.updated": [open_interest],
            "analytics.open_interest.anomaly": [open_interest],
            "analytics.open_interest.capitulation": [open_interest],
            "analytics.open_interest.divergence": [open_interest],
            "analytics.open_interest.regime_changed": [open_interest],
            "analytics.open_interest.squeeze_setup": [open_interest],

            # Optional domains for future datasets.
            "analytics.orderflow.updated": [orderflow],
            "analytics.liquidations.updated": [liquidations],
            "analytics.spreads.updated": [spreads],
            "analytics.spread.updated": [spreads],
        }

        for topic, categories in topic_map.items():
            event_to_categories.setdefault(topic, categories)

        # Optional safety net for local debugging. Disabled by default because it
        # routes every analytics event into broad strategy selection.
        if env_bool("BACKTEST_STRATEGY_SUBSCRIBE_ALL_ANALYTICS", False):
            event_to_categories.setdefault(
                "analytics.*",
                [price_action, funding, open_interest, orderflow, liquidity, liquidations, spreads],
            )

        self.diagnostics.loaded.append(
            "strategy.routing.event_to_categories patched for backtest analytics topics"
        )

    def build_strategy_registry(
        self,
        strategy_config: Any,
        *,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> Any:
        """
        Build production StrategyRegistry with data-aware concrete strategy
        factories.

        The strategy package owns routing/evaluation/building. This bootstrap
        only decides which concrete strategy classes are available for the
        current backtest streams.

        Example:
        - candles=True -> price_action strategies
        - funding=True -> funding strategies
        - open_interest=True -> open_interest strategies
        - trades=True -> orderflow + whales
        - orderbook=True -> liquidity + spoofing
        - liquidations=True -> liquidations
        - spreads=True -> spreads
        - hybrid=True only when enough enabled domains exist
        """

        candidates = [
            "strategy.presets.build_default_strategy_registry",
            "strategy.registry.StrategyRegistry",
        ]

        strategy_factories = self.build_strategy_factories(strategy_config)

        available = {
            "config": strategy_config,
            "strategy_config": strategy_config,
            "event_bus": event_bus,
            "scheduler": scheduler,
            "strategy_factories": strategy_factories,
            "strict": env_bool("BACKTEST_STRATEGY_STRICT_FACTORIES", False),
            "replace": True,
            "emit_events": True,
        }

        registry = self.instantiate_first_required(
            "strategy registry",
            candidates,
            available,
        )

        total = self.strategy_registry_count(registry)
        if total <= 0:
            raise BacktestDependencyError(
                "Strategy registry is empty after bootstrap. "
                "No concrete strategies were registered for the enabled streams.\n"
                f"{self.diagnostics.format()}"
            )

        self.diagnostics.loaded.append(
            f"[strategy] StrategyRegistry populated with {total} strategies"
        )
        return registry

    def build_strategy_factories(self, strategy_config: Any) -> dict[str, Any]:
        """
        Build data-aware strategy factory map.

        This method does not auto-discover arbitrary classes. It uses explicit
        candidate paths and only enables domains supported by current backtest
        streams. Missing required-domain factories are kept in diagnostics and
        later validated by validate_strategy_registry().
        """

        allowed_names = self.strategy_names_from_config(strategy_config)
        requested_names = self.filter_strategy_names_for_enabled_streams(allowed_names)

        candidates_by_name = self.strategy_factory_paths()
        factories: dict[str, Any] = {}

        for name in requested_names:
            candidates = candidates_by_name.get(name)
            if not candidates:
                self.diagnostics.add_failure(
                    f"strategy_factory:{name}",
                    "No explicit factory path configured in backtesting bootstrap.",
                )
                continue

            factory, path = self.import_first_strategy_candidate(candidates)
            if factory is None:
                self.diagnostics.add_failure(
                    f"strategy_factory:{name}",
                    "Could not import any candidate: " + ", ".join(candidates),
                )
                continue

            factories[name] = factory
            self.diagnostics.loaded.append(f"[strategy_factory] {name} -> {path}")

        return factories

    def strategy_names_from_config(self, strategy_config: Any) -> list[str]:
        """
        Return enabled strategy names from StrategyConfig / PresetConfig.

        Prefer preset.enabled_strategy_names when present because
        build_default_strategy_registry uses the same rule. Fall back to
        config.strategies keys.
        """

        preset = getattr(strategy_config, "preset", None)
        enabled = getattr(preset, "enabled_strategy_names", None)
        if enabled:
            return [str(name).strip() for name in enabled if str(name).strip()]

        strategies = getattr(strategy_config, "strategies", None)
        if isinstance(strategies, dict):
            return [str(name).strip() for name in strategies if str(name).strip()]

        return []

    def filter_strategy_names_for_enabled_streams(self, names: Sequence[str]) -> list[str]:
        domain_map = self.strategy_domain_by_name()
        enabled_domains = self.enabled_strategy_domains()

        result: list[str] = []
        skipped: list[str] = []

        for raw_name in names:
            name = str(raw_name).strip()
            if not name:
                continue

            domain = domain_map.get(name)
            if domain is None:
                # Unknown catalog entry: do not auto-enable in backtesting.
                self.diagnostics.add_skipped(
                    f"strategy_factory:{name}",
                    "strategy domain is unknown to backtesting bootstrap",
                )
                continue

            if domain in enabled_domains:
                result.append(name)
            else:
                skipped.append(name)

        if skipped:
            self.diagnostics.add_skipped(
                "strategy_factories",
                "Skipped strategies not supported by enabled streams: "
                + ", ".join(sorted(skipped)),
            )

        return list(dict.fromkeys(result))

    def enabled_strategy_domains(self) -> set[str]:
        domains: set[str] = set()

        if self.config.enable_candles:
            domains.add("price_action")

        if self.config.enable_funding:
            domains.add("funding")

        if self.config.enable_open_interest:
            domains.add("open_interest")

        if self.config.enable_trades:
            domains.add("orderflow")
            domains.add("whales")

        if self.config.enable_orderbook:
            domains.add("liquidity")
            domains.add("spoofing")

        if self.config.enable_liquidations:
            domains.add("liquidations")

        if self.config.enable_spreads:
            domains.add("spreads")

        # Hybrid strategies are useful only when several analytics domains are
        # present. This avoids selecting whale/orderflow hybrids on a
        # candles+funding+OI dataset.
        enable_hybrid = env_bool("BACKTEST_STRATEGY_ENABLE_HYBRID", True)
        min_hybrid_domains = env_int("BACKTEST_STRATEGY_MIN_HYBRID_DOMAINS", 3)
        hybrid_base_domains = {
            "price_action",
            "funding",
            "open_interest",
            "orderflow",
            "whales",
            "liquidity",
            "liquidations",
            "spoofing",
            "spreads",
        }
        if enable_hybrid and len(domains.intersection(hybrid_base_domains)) >= min_hybrid_domains:
            domains.add("hybrid")

        return domains

    @staticmethod
    def strategy_domain_by_name() -> dict[str, str]:
        return {
            # price_action
            "market_structure": "price_action",
            "fvg_reaction": "price_action",
            "support_resistance_reaction": "price_action",
            "trend_continuation": "price_action",

            # open_interest
            "oi_divergence": "open_interest",
            "oi_breakout_confirmation": "open_interest",
            "oi_anomaly": "open_interest",
            "oi_capitulation": "open_interest",

            # funding
            "funding_extreme_reversal": "funding",
            "funding_divergence": "funding",

            # orderflow
            "cvd_divergence": "orderflow",
            "orderflow_continuation": "orderflow",
            "orderflow_reversal": "orderflow",

            # whales
            "whale_accumulation": "whales",
            "whale_distribution": "whales",
            "whale_breakout": "whales",

            # liquidity
            "liquidity_sweep": "liquidity",
            "stop_hunt_reversal": "liquidity",
            "equal_high_low": "liquidity",

            # liquidations
            "liquidation_cascade": "liquidations",
            "squeeze_reversal": "liquidations",

            # spoofing
            "spoofing_reversal": "spoofing",
            "fake_liquidity_trap": "spoofing",

            # spreads
            "cross_exchange_arb": "spreads",
            "spot_futures_basis": "spreads",
            "spread_momentum": "spreads",
            "funding_adjusted_basis": "spreads",

            # hybrid
            "confluence": "hybrid",
            "hybrid_confluence": "hybrid",
            "trend_stack": "hybrid",
            "mean_reversion_stack": "hybrid",
            "liquidation_whale": "hybrid",
            "whale_orderflow_breakout": "hybrid",
        }

    @staticmethod
    def strategy_factory_paths() -> dict[str, tuple[str, ...]]:
        """
        Known concrete strategy class candidates keyed by StrategyCatalog name.

        Include both current unified strategy paths and a few legacy class-name
        aliases where harmless. Import failures stay diagnostic, not silent.
        """

        return {
            # -----------------------------------------------------------------
            # price_action
            # -----------------------------------------------------------------
            "market_structure": (
                "strategy.strategies.price_action.market_structure_strategy.MarketStructureStrategy",
            ),
            "fvg_reaction": (
                "strategy.strategies.price_action.fvg_reaction_strategy.FVGReactionStrategy",
                "strategy.strategies.price_action.fvg_reaction_strategy.FvgReactionStrategy",
            ),
            "support_resistance_reaction": (
                "strategy.strategies.price_action.support_resistance_reaction_strategy.SupportResistanceReactionStrategy",
            ),
            "trend_continuation": (
                "strategy.strategies.price_action.trend_continuation_strategy.TrendContinuationStrategy",
            ),

            # -----------------------------------------------------------------
            # open_interest
            # -----------------------------------------------------------------
            "oi_divergence": (
                "strategy.strategies.open_interest.oi_divergence_strategy.OIDivergenceStrategy",
                "strategy.strategies.open_interest.OIDivergenceStrategy",
            ),
            "oi_breakout_confirmation": (
                "strategy.strategies.open_interest.oi_breakout_confirmation_strategy.OIBreakoutConfirmationStrategy",
                "strategy.strategies.open_interest.OIBreakoutConfirmationStrategy",
            ),
            "oi_anomaly": (
                "strategy.strategies.open_interest.oi_anomaly_strategy.OIAnomalyStrategy",
                "strategy.strategies.open_interest.OIAnomalyStrategy",
            ),
            "oi_capitulation": (
                "strategy.strategies.open_interest.oi_capitulation_strategy.OICapitulationStrategy",
                "strategy.strategies.open_interest.OICapitulationStrategy",
            ),

            # -----------------------------------------------------------------
            # funding
            # -----------------------------------------------------------------
            "funding_extreme_reversal": (
                "strategy.strategies.funding.funding_extreme_reversal_strategy.FundingExtremeReversalStrategy",
                "strategy.strategies.funding.FundingExtremeReversalStrategy",
            ),
            "funding_divergence": (
                "strategy.strategies.funding.funding_divergence_strategy.FundingDivergenceStrategy",
                "strategy.strategies.funding.FundingDivergenceStrategy",
            ),

            # -----------------------------------------------------------------
            # orderflow
            # -----------------------------------------------------------------
            "cvd_divergence": (
                "strategy.strategies.orderflow.cvd_divergence_strategy.CVDDivergenceStrategy",
                "strategy.strategies.orderflow.cvd_divergence_strategy.CvdDivergenceStrategy",
            ),
            "orderflow_continuation": (
                "strategy.strategies.orderflow.orderflow_continuation_strategy.OrderflowContinuationStrategy",
                "strategy.strategies.orderflow.orderflow_continuation_strategy.OrderFlowContinuationStrategy",
            ),
            "orderflow_reversal": (
                "strategy.strategies.orderflow.orderflow_reversal_strategy.OrderflowReversalStrategy",
                "strategy.strategies.orderflow.orderflow_reversal_strategy.OrderFlowReversalStrategy",
            ),

            # -----------------------------------------------------------------
            # whales
            # -----------------------------------------------------------------
            "whale_accumulation": (
                "strategy.strategies.whales.whale_accumulation_strategy.WhaleAccumulationStrategy",
            ),
            "whale_distribution": (
                "strategy.strategies.whales.whale_distribution_strategy.WhaleDistributionStrategy",
            ),
            "whale_breakout": (
                "strategy.strategies.whales.whale_breakout_strategy.WhaleBreakoutStrategy",
            ),

            # -----------------------------------------------------------------
            # liquidity
            # -----------------------------------------------------------------
            "liquidity_sweep": (
                "strategy.strategies.liquidity.liquidity_sweep_strategy.LiquiditySweepStrategy",
            ),
            "stop_hunt_reversal": (
                "strategy.strategies.liquidity.stop_hunt_reversal_strategy.StopHuntReversalStrategy",
            ),
            "equal_high_low": (
                "strategy.strategies.liquidity.equal_high_low_strategy.EqualHighLowStrategy",
            ),

            # -----------------------------------------------------------------
            # liquidations
            # -----------------------------------------------------------------
            "liquidation_cascade": (
                "strategy.strategies.liquidations.liquidation_cascade_strategy.LiquidationCascadeStrategy",
            ),
            "squeeze_reversal": (
                "strategy.strategies.liquidations.squeeze_reversal_strategy.SqueezeReversalStrategy",
            ),

            # -----------------------------------------------------------------
            # spoofing
            # -----------------------------------------------------------------
            "spoofing_reversal": (
                "strategy.strategies.spoofing.spoofing_reversal_strategy.SpoofingReversalStrategy",
            ),
            "fake_liquidity_trap": (
                "strategy.strategies.spoofing.fake_liquidity_trap_strategy.FakeLiquidityTrapStrategy",
            ),

            # -----------------------------------------------------------------
            # spreads
            # -----------------------------------------------------------------
            "cross_exchange_arb": (
                "strategy.strategies.spreads.cross_exchange_arb_strategy.CrossExchangeArbStrategy",
                "strategy.strategies.spreads.cross_exchange_arb_strategy.CrossExchangeArbitrageStrategy",
            ),
            "spot_futures_basis": (
                "strategy.strategies.spreads.spot_futures_basis_strategy.SpotFuturesBasisStrategy",
            ),
            "spread_momentum": (
                "strategy.strategies.spreads.spread_momentum_strategy.SpreadMomentumStrategy",
            ),
            "funding_adjusted_basis": (
                "strategy.strategies.spreads.funding_adjusted_basis_strategy.FundingAdjustedBasisStrategy",
            ),

            # -----------------------------------------------------------------
            # hybrid
            # -----------------------------------------------------------------
            "confluence": (
                "strategy.strategies.hybrid.confluence_strategy.ConfluenceStrategy",
            ),
            "hybrid_confluence": (
                "strategy.strategies.hybrid.confluence_strategy.ConfluenceStrategy",
            ),
            "trend_stack": (
                "strategy.strategies.hybrid.trend_stack_strategy.TrendStackStrategy",
            ),
            "mean_reversion_stack": (
                "strategy.strategies.hybrid.mean_reversion_stack_strategy.MeanReversionStackStrategy",
            ),
            "liquidation_whale": (
                "strategy.strategies.hybrid.liquidation_whale_strategy.LiquidationWhaleStrategy",
            ),
            "whale_orderflow_breakout": (
                "strategy.strategies.hybrid.whale_orderflow_breakout_strategy.WhaleOrderflowBreakoutStrategy",
            ),
        }

    @staticmethod
    def import_first_strategy_candidate(paths: Sequence[str]) -> tuple[Any | None, str | None]:
        for path in paths:
            target = import_object_or_none(path)
            if target is not None:
                return target, path
        return None, None

    @staticmethod
    def strategy_registry_count(registry: Any) -> int:
        count = getattr(registry, "count", None)
        if callable(count):
            try:
                return int(count())
            except (TypeError, ValueError, AttributeError):
                pass

        list_all = getattr(registry, "list_all", None)
        if callable(list_all):
            try:
                return len(list_all())
            except (TypeError, ValueError, AttributeError):
                pass

        raw = getattr(registry, "_strategies", None)
        if isinstance(raw, dict):
            return len(raw)

        return 0

    @staticmethod
    def strategy_registry_categories(registry: Any) -> set[str]:
        by_category = getattr(registry, "_by_category", None)
        categories: set[str] = set()

        if isinstance(by_category, dict):
            for category, values in by_category.items():
                try:
                    if len(values) <= 0:
                        continue
                except TypeError:
                    pass
                categories.add(str(getattr(category, "value", category)))
            return categories

        list_all = getattr(registry, "list_all", None)
        if callable(list_all):
            try:
                for strategy in list_all():
                    category = getattr(strategy, "category", None)
                    if category is not None:
                        categories.add(str(getattr(category, "value", category)))
            except (TypeError, ValueError, AttributeError):
                pass

        return categories

    def validate_strategy_registry(self, registry: Any) -> None:
        """
        Validate that enabled analytics streams have matching strategy domains.
        """

        categories = self.strategy_registry_categories(registry)
        missing: list[str] = []

        if self.config.enable_candles and "price_action" not in categories:
            missing.append("PRICE_ACTION strategies for candle/price_action analytics")

        if self.config.enable_open_interest and "open_interest" not in categories:
            missing.append("OPEN_INTEREST strategies for analytics.oi.*")

        if self.config.enable_funding and "funding" not in categories:
            missing.append("FUNDING strategies for analytics.funding.*")

        if self.config.enable_trades:
            if "orderflow" not in categories:
                missing.append("ORDERFLOW strategies for trade/orderflow analytics")
            if "whales" not in categories:
                missing.append("WHALES strategies for whale/trades analytics")

        if self.config.enable_orderbook:
            if "liquidity" not in categories:
                missing.append("LIQUIDITY strategies for orderbook/liquidity analytics")
            if "spoofing" not in categories:
                missing.append("SPOOFING strategies for orderbook/spoofing analytics")

        if self.config.enable_liquidations and "liquidations" not in categories:
            missing.append("LIQUIDATIONS strategies for liquidation analytics")

        if self.config.enable_spreads and "spreads" not in categories:
            missing.append("SPREADS strategies for spread analytics")

        if not missing:
            return

        raise BacktestDependencyError(
            "BacktestProjectBootstrap registry is missing strategy domains required "
            "for enabled analytics streams:\n- "
            + "\n- ".join(missing)
            + "\n\n"
            + self.diagnostics.format()
        )

    @staticmethod
    def build_strategy_runtime_state(strategy_config: Any) -> Any:
        state_cls = import_required("strategy.state.StrategyRuntimeState")
        return construct_with_signature(
            state_cls,
            {
                "config": strategy_config,
                "strategy_config": strategy_config,
            },
        )

    def build_signal_processor(
        self,
        *,
        strategy_config: Any,
        registry: Any,
        runtime_state: Any,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> Any:
        cls = import_required("strategy.processor.SignalProcessor")
        return construct_with_signature(
            cls,
            {
                "config": strategy_config,
                "strategy_config": strategy_config,
                "registry": registry,
                "strategy_registry": registry,
                "runtime_state": runtime_state,
                "state": runtime_state,
                "event_bus": event_bus,
                "scheduler": scheduler,
            },
        )

    def build_strategy_engine(
        self,
        *,
        strategy_config: Any,
        registry: Any,
        runtime_state: Any,
        signal_processor: Any,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> Any:
        cls = import_required("strategy.engine.StrategyEngine")
        return construct_with_signature(
            cls,
            {
                "config": strategy_config,
                "strategy_config": strategy_config,
                "registry": registry,
                "strategy_registry": registry,
                "runtime_state": runtime_state,
                "state": runtime_state,
                "signal_processor": signal_processor,
                "processor": signal_processor,
                "event_bus": event_bus,
                "scheduler": scheduler,
            },
        )

    # ------------------------------------------------------------------
    # Risk layer
    # ------------------------------------------------------------------

    def build_risk_manager(
        self,
        *,
        core_config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> Any:
        """
        Build production RiskManager.

        RiskManager must listen to signal.* and execution./position. events and
        emit signal.confirmed / risk.*. Bootstrap does not approve signals.
        """

        risk_config = self.build_risk_config(core_config)
        cls = import_required("risk.risk_manager.RiskManager")
        return construct_with_signature(
            cls,
            {
                "config": risk_config,
                "risk_config": risk_config,
                "app_config": core_config,
                "core_config": core_config,
                "event_bus": event_bus,
                "scheduler": scheduler,
            },
        )

    @staticmethod
    def build_risk_config(core_config: Config) -> Any:
        risk_config_cls = import_object_or_none("risk.config.RiskConfig")
        if risk_config_cls is None:
            return getattr(core_config, "risk", None)

        existing = getattr(core_config, "risk", None)
        if isinstance(existing, risk_config_cls):
            return existing

        return construct_with_signature(
            risk_config_cls,
            {
                "core_config": core_config,
                "app_config": core_config,
            },
        )

    # ------------------------------------------------------------------
    # Generic component helpers
    # ------------------------------------------------------------------

    def instantiate_required(self, path: str, available: dict[str, Any]) -> Any:
        target = import_required(path)

        try:
            if inspect.isclass(target):
                return construct_with_signature(target, available)
            return target(**filter_callable_kwargs(target, available))
        except (TypeError, ValueError, AttributeError) as exc:
            raise BacktestDependencyError(
                f"Required component could not be instantiated: {path}.",
                details={"reason": str(exc)},
            ) from exc

    def instantiate_optional(self, path: str, available: dict[str, Any]) -> Any | None:
        target = import_object_or_none(path)
        if target is None:
            self.diagnostics.add_failure(path, "import failed")
            return None

        try:
            if inspect.isclass(target):
                return construct_with_signature(target, available)
            return target(**filter_callable_kwargs(target, available))
        except (TypeError, ValueError, AttributeError) as exc:
            self.diagnostics.add_failure(path, f"{exc.__class__.__name__}: {exc}")
            return None

    def instantiate_first_required(
        self,
        label: str,
        candidates: Sequence[str],
        available: dict[str, Any],
    ) -> Any:
        errors: list[str] = []

        for path in candidates:
            target = import_object_or_none(path)
            if target is None:
                errors.append(f"{path}: import failed")
                continue

            try:
                if inspect.isclass(target):
                    return construct_with_signature(target, available)
                return target(**filter_callable_kwargs(target, available))
            except (TypeError, ValueError, AttributeError) as exc:
                errors.append(f"{path}: {exc}")

        raise BacktestDependencyError(
            f"Could not build {label}.",
            details={"errors": errors},
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_pipeline(
        self,
        *,
        data_caches: Sequence[Any],
        analytics_components: Sequence[Any],
        strategy_pipeline: BuiltStrategyPipeline,
        risk_manager: Any,
    ) -> None:
        if not data_caches:
            raise BacktestDependencyError("Backtest bootstrap built no data caches.")

        if not analytics_components and self.config.require_analytics_for_enabled_streams:
            raise BacktestDependencyError(
                "Backtest bootstrap built no analytics components.\n"
                f"{self.diagnostics.format()}"
            )

        if strategy_pipeline.strategy_engine is None:
            raise BacktestDependencyError("StrategyEngine is required for full-pipeline backtest.")

        if strategy_pipeline.signal_processor is None:
            raise BacktestDependencyError("SignalProcessor is required for full-pipeline backtest.")

        if risk_manager is None:
            raise BacktestDependencyError("RiskManager is required for full-pipeline backtest.")


# =============================================================================
# Utility functions
# =============================================================================


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if not normalized:
        return default

    return normalized in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def unique_strings(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)

    return result


def qualified_name(obj: Any) -> str:
    module = getattr(obj, "__module__", "")
    name = getattr(obj, "__qualname__", getattr(obj, "__name__", obj.__class__.__name__))
    if module:
        return f"{module}.{name}"
    return str(name)


def import_required(path: str) -> Any:
    obj = import_object_or_none(path)
    if obj is None:
        raise BacktestDependencyError(
            "Required object could not be imported.",
            details={"path": path},
        )
    return obj


def import_object_or_none(path: str) -> Any | None:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        return None

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    return getattr(module, attr_name, None)


def filter_callable_kwargs(callable_obj: Any, available: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}

    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return {key: value for key, value in available.items() if value is not None}

    kwargs: dict[str, Any] = {}

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue

        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue

        if name in available and available[name] is not None:
            kwargs[name] = available[name]

    return kwargs


def construct_with_signature(cls: type[Any], available: dict[str, Any]) -> Any:
    """
    Instantiate class using only parameters accepted by its constructor.
    """

    kwargs = filter_callable_kwargs(cls, available)
    return cls(**kwargs)


def dedupe_components(components: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()

    for component in components:
        key = qualified_name(component.__class__)
        if key in seen:
            continue
        result.append(component)
        seen.add(key)

    return result


def find_component(components: Sequence[Any], class_name: str) -> Any | None:
    for component in components:
        if component.__class__.__name__ == class_name:
            return component
    return None


def find_named_value(values: dict[str, Any], class_name: str) -> Any | None:
    for value in values.values():
        if value is not None and value.__class__.__name__ == class_name:
            return value
    return None


def print_pipeline_summary(pipeline: BuiltBacktestPipeline) -> None:
    """
    Optional helper for run.py.
    """

    print()
    print("========== PIPELINE ==========")
    print("Data caches:")
    for component in pipeline.data_caches:
        print(f"- {component.__class__.__name__}")

    print("Analytics:")
    for component in pipeline.analytics_components:
        print(f"- {component.__class__.__name__}")

    print(f"Strategy registry: {pipeline.strategy_registry.__class__.__name__}")
    print(f"Signal processor:  {pipeline.signal_processor.__class__.__name__}")
    print(f"Strategy engine:   {pipeline.strategy_engine.__class__.__name__}")
    print(f"Risk manager:      {pipeline.risk_manager.__class__.__name__}")
    print("==============================")