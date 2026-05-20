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
        event_bus = self.event_bus or self.build_event_bus(core_config)
        scheduler = self.scheduler or self.build_scheduler(event_bus)

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

    @staticmethod
    def build_event_bus(core_config: Config) -> EventBus:
        return construct_with_signature(
            EventBus,
            {
                "config": getattr(core_config, "event_bus", None),
                "event_bus_config": getattr(core_config, "event_bus", None),
            },
        )

    @staticmethod
    def build_scheduler(event_bus: EventBus) -> Scheduler:
        return construct_with_signature(
            Scheduler,
            {
                "event_bus": event_bus,
            },
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
        registry = self.build_strategy_registry(strategy_config)
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

    def build_strategy_registry(self, strategy_config: Any) -> Any:
        candidates = [
            "strategy.presets.build_default_strategy_registry",
            "strategy.registry.StrategyRegistry",
        ]

        available = {
            "config": strategy_config,
            "strategy_config": strategy_config,
            "event_bus": self.event_bus,
            "scheduler": self.scheduler,
        }

        return self.instantiate_first_required("strategy registry", candidates, available)

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