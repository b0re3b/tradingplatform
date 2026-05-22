"""
main.py — Production entry point for the async futures trading system.

Architecture layers (EventBus-first, no direct cross-layer calls):

  exchanges/*  →  market.*  →  data/*_cache  →  market.*.updated
               →  analytics.*  →  strategy (signal.generated)
               →  risk (signal.confirmed)  →  execution (orders/positions)

  ai/news_service  — runs in parallel, publishes news.* events independently.

Usage:
    python main.py [--symbols BTCUSDT ETHUSDT] [--exchange binance] [--env .env]
    python main.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Core (must be first — logger init before any other import)
# ---------------------------------------------------------------------------
from core.config import Config
from core.event_bus import EventBus, QueueFullPolicy
from core.logger import get_logger, init_logger
from core.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
from data.candles_cache import CandlesCache
from data.funding_cache import FundingCache
from data.market_stream import MarketStream, MarketStreamConfig
from data.open_interest_cache import OpenInterestCache
from data.orderbook_cache import OrderBookCache
from data.trades_cache import TradesCache

# ---------------------------------------------------------------------------
# Exchange adapters
# ---------------------------------------------------------------------------
from exchanges.binance.binance_rest import BinanceRestClient
from exchanges.binance.binance_ws import BinanceWebSocketClient, BinanceWebSocketClientConfig

# ---------------------------------------------------------------------------
# Analytics layer
# ---------------------------------------------------------------------------
from analytics.funding.funding_analyzer import FundingAnalyzer
from analytics.liquidations.config import LiquidationStreamConfig, LiquidationsConfig
from analytics.liquidations.liquidation_stream import LiquidationStream
from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidity.liquidity_service import LiquidityService
from analytics.open_interest.oi_analyzer import OIAnalyzer
from analytics.orderflow.analyzer import OrderFlowAnalyzer
from analytics.price_action.price_action_analyzer import PriceActionAnalyzer, PriceActionAnalyzerConfig
from analytics.spoofing.analyzer import SpoofingAnalyzer
from analytics.spoofing.config import SpoofingConfig
from analytics.spreads.spread_analyzer import SpreadAnalyzer
from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.config import WhalesConfig

# ---------------------------------------------------------------------------
# Strategy layer
# ---------------------------------------------------------------------------
from strategy.config import StrategyConfig
from strategy.engine import StrategyEngine
from strategy.strategies.funding.funding_divergence_strategy import FundingDivergenceStrategy
from strategy.strategies.funding.funding_extreme_reversal_strategy import FundingExtremeReversalStrategy
from strategy.strategies.hybrid.confluence_strategy import ConfluenceStrategy
from strategy.strategies.hybrid.liquidation_whale_strategy import LiquidationWhaleStrategy
from strategy.strategies.hybrid.liquidity_orderflow_reversal_strategy import LiquidityOrderflowReversalStrategy
from strategy.strategies.hybrid.mean_reversion_stack_strategy import MeanReversionStackStrategy
from strategy.strategies.hybrid.oi_funding_squeeze_strategy import OIFundingSqueezeStrategy
from strategy.strategies.hybrid.trend_stack_strategy import TrendStackStrategy
from strategy.strategies.hybrid.whale_orderflow_breakout_strategy import WhaleOrderflowBreakoutStrategy
from strategy.strategies.liquidations.liquidation_cascade_strategy import LiquidationCascadeStrategy
from strategy.strategies.liquidations.squeeze_reversal_strategy import SqueezeReversalStrategy
from strategy.strategies.liquidity.equal_high_low_strategy import EqualHighLowStrategy
from strategy.strategies.liquidity.liquidity_map_bias_strategy import LiquidityMapBiasStrategy
from strategy.strategies.liquidity.liquidity_sweep_strategy import LiquiditySweepStrategy
from strategy.strategies.liquidity.stop_hunt_reversal_strategy import StopHuntReversalStrategy
from strategy.strategies.open_interest.oi_anomaly_strategy import OIAnomalyStrategy
from strategy.strategies.open_interest.oi_breakout_confirmation_strategy import OIBreakoutConfirmationStrategy
from strategy.strategies.open_interest.oi_capitulation_strategy import OICapitulationStrategy
from strategy.strategies.open_interest.oi_divergence_strategy import OIDivergenceStrategy
from strategy.strategies.orderflow.cvd_divergence_strategy import CvdDivergenceStrategy
from strategy.strategies.orderflow.orderflow_continuation_strategy import OrderflowContinuationStrategy
from strategy.strategies.orderflow.orderflow_reversal_strategy import OrderflowReversalStrategy
from strategy.strategies.price_action.fvg_reaction_strategy import FVGReactionStrategy
from strategy.strategies.price_action.market_structure_strategy import MarketStructureStrategy
from strategy.strategies.price_action.support_resistance_reaction_strategy import SupportResistanceReactionStrategy
from strategy.strategies.price_action.trend_continuation_strategy import TrendContinuationStrategy
from strategy.strategies.spoofing.composite_spoofing_strategy import CompositeSpoofingStrategy
from strategy.strategies.spoofing.fake_liquidity_trap_strategy import FakeLiquidityTrapStrategy
from strategy.strategies.spoofing.layering_trap_strategy import LayeringTrapStrategy
from strategy.strategies.spoofing.order_pull_reversal_strategy import OrderPullReversalStrategy
from strategy.strategies.spoofing.pressure_bluff_reversal_strategy import PressureBluffReversalStrategy
from strategy.strategies.spoofing.spoofing_absorption_reversal_strategy import SpoofingAbsorptionReversalStrategy
from strategy.strategies.spoofing.spoofing_reversal_strategy import SpoofingReversalStrategy
from strategy.strategies.spreads.cross_exchange_arb_strategy import CrossExchangeArbStrategy
from strategy.strategies.spreads.funding_adjusted_basis_strategy import FundingAdjustedBasisStrategy
from strategy.strategies.spreads.spot_futures_basis_strategy import SpotFuturesBasisStrategy
from strategy.strategies.spreads.spread_mean_reversion_strategy import SpreadMeanReversionStrategy
from strategy.strategies.spreads.spread_momentum_strategy import SpreadMomentumStrategy
from strategy.strategies.whales.whale_absorption_strategy import WhaleAbsorptionStrategy
from strategy.strategies.whales.whale_accumulation_strategy import WhaleAccumulationStrategy
from strategy.strategies.whales.whale_breakout_strategy import WhaleBreakoutStrategy
from strategy.strategies.whales.whale_distribution_strategy import WhaleDistributionStrategy
from strategy.strategies.whales.whale_liquidation_reversal_strategy import WhaleLiquidationReversalStrategy

# ---------------------------------------------------------------------------
# Risk layer
# ---------------------------------------------------------------------------
from risk.config import RiskConfig
from risk.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Execution layer
# ---------------------------------------------------------------------------
from execution.config import (
    ExecutionConfig,
    OrderManagerConfig,
    PositionManagerConfig,
    SLTPManagerConfig,
    SmartExecutionConfig,
    TradeExecutorConfig,
)
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from execution.sl_tp_manager import SLTPManager
from execution.smart_execution import SmartExecution
from execution.trade_executor import TradeExecutor

# ---------------------------------------------------------------------------
# AI / News layer (parallel, non-trading)
# ---------------------------------------------------------------------------
from ai.config import build_default_news_ai_config
from ai.news_service import NewsAIService


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async futures trading system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Trading symbols (USD-M futures)",
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        choices=["binance"],
        help="Primary exchange",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Primary candle timeframe",
    )
    parser.add_argument(
        "--news",
        action="store_true",
        default=True,
        help="Enable AI news service (parallel, does not affect trading)",
    )
    parser.add_argument(
        "--no-news",
        dest="news",
        action="store_false",
        help="Disable AI news service",
    )
    return parser.parse_args()


# ============================================================================
# System bootstrap
# ============================================================================

class TradingSystem:
    """
    Root orchestrator.

    Owns all service lifecycles.
    Does NOT contain any trading logic — that lives in strategy/risk/execution.
    Does NOT directly wire analytics → strategy → risk → execution.
    All communication goes through EventBus.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        exchange: str,
        timeframe: str,
        enable_news: bool,
        env_file: str,
    ) -> None:
        self._symbols = symbols
        self._exchange = exchange
        self._timeframe = timeframe
        self._enable_news = enable_news
        self._env_file = env_file

        self._running = False
        self._shutdown_event = asyncio.Event()

        # populated during build()
        self._config: Config | None = None
        self._event_bus: EventBus | None = None
        self._scheduler: Scheduler | None = None
        self._services: list[Any] = []
        self._logger = get_logger(__name__, event_type="trading_system")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_config(self) -> Config:
        return Config.from_env(self._env_file)

    def _build_event_bus(self, config: Config) -> EventBus:
        return EventBus(
            max_queue_size=config.event_bus.max_queue_size,
            worker_count=config.event_bus.worker_count,
            queue_full_policy=QueueFullPolicy(config.event_bus.queue_full_policy),
            max_retries=config.event_bus.max_retries,
            retry_delay=config.event_bus.retry_delay,
            enable_metrics=config.event_bus.enable_metrics,
            service_name="event_bus",
        )

    def _build_scheduler(self, config: Config, event_bus: EventBus) -> Scheduler:
        return Scheduler(
            event_bus=event_bus,
            tick_interval=config.scheduler.tick_interval,
            service_name="scheduler",
        )

    # ------ Data layer --------------------------------------------------

    def _build_data_layer(
        self,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> tuple[CandlesCache, OrderBookCache, TradesCache, FundingCache, OpenInterestCache]:
        candles_cache = CandlesCache(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        orderbook_cache = OrderBookCache(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        trades_cache = TradesCache(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        funding_cache = FundingCache(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        oi_cache = OpenInterestCache(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        return candles_cache, orderbook_cache, trades_cache, funding_cache, oi_cache

    # ------ Exchange adapters -------------------------------------------

    def _build_exchange_layer(
        self,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> dict[str, Any]:
        ws_config = BinanceWebSocketClientConfig.from_core_config(
            config=config,
            symbols=self._symbols,
            streams=["trade", "depth", "kline", "markPrice", "openInterest"],
            depth_level="20",
            kline_interval=self._timeframe,
            enable_private_stream=bool(config.exchange.credentials.api_key),
        )
        ws_client = BinanceWebSocketClient(
            config=config,
            event_bus=event_bus,
            ws_config=ws_config,
            scheduler=scheduler,
        )
        rest_client = BinanceRestClient(
            config=config,
            event_bus=event_bus,
        )
        return {
            "binance_ws": ws_client,
            "binance_rest": rest_client,
        }

    # ------ Analytics layer --------------------------------------------

    def _build_analytics_layer(
        self,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        trades_cache: TradesCache,
        orderbook_cache: OrderBookCache,
    ) -> list[Any]:
        """
        Build all analytics analyzers.
        Each receives EventBus and Scheduler only.
        They subscribe to market.*.updated topics and publish analytics.* topics.
        No direct calls to strategy/risk/execution.
        """
        analyzers: list[Any] = []

        # Funding
        funding_analyzer = FundingAnalyzer(
            event_bus=event_bus,
            scheduler=scheduler,
        )
        analyzers.append(funding_analyzer)

        # Open Interest
        oi_analyzer = OIAnalyzer(
            event_bus=event_bus,
            scheduler=scheduler,
        )
        analyzers.append(oi_analyzer)

        # Orderflow (needs cache read access)
        orderflow_analyzer = OrderFlowAnalyzer(
            event_bus=event_bus,
            trades_cache=trades_cache,
            orderbook_cache=orderbook_cache,
            scheduler=scheduler,
            default_exchange=self._exchange,
            default_market_type="usdm_futures",
        )
        analyzers.append(orderflow_analyzer)

        # Liquidations
        liq_stream_config = LiquidationStreamConfig(
            default_exchange=self._exchange,
            default_market_type="usdm_futures",
        )
        liquidation_stream = LiquidationStream(
            event_bus=event_bus,
            config=liq_stream_config,
            scheduler=scheduler,
        )
        analyzers.append(liquidation_stream)

        # Liquidity
        liquidity_config = LiquidityConfig()
        liquidity_map = LiquidityMap(config=liquidity_config)
        liquidity_service = LiquidityService(
            event_bus=event_bus,
            scheduler=scheduler,
            config=liquidity_config,
            liquidity_map=liquidity_map,
        )
        analyzers.append(liquidity_service)

        # Price Action — one instance per symbol
        for symbol in self._symbols:
            pa_analyzer = PriceActionAnalyzer(
                symbol=symbol,
                timeframe=self._timeframe,
                event_bus=event_bus,
                exchange=self._exchange,
                market_type="usdm_futures",
                scheduler=scheduler,
            )
            analyzers.append(pa_analyzer)

        # Spoofing
        spoofing_config = SpoofingConfig()
        spoofing_analyzer = SpoofingAnalyzer(
            event_bus=event_bus,
            scheduler=scheduler,
            config=spoofing_config,
            orderbook_cache=orderbook_cache,
        )
        analyzers.append(spoofing_analyzer)

        # Spreads (spot/futures basis + cross-exchange)
        spread_analyzer = SpreadAnalyzer(
            event_bus=event_bus,
            scheduler=scheduler,
            enable_spot_futures=True,
            enable_cross_exchange=True,
        )
        analyzers.append(spread_analyzer)

        # Whales
        whales_config = WhalesConfig()
        whale_analyzer = WhaleAnalyzer(
            config=whales_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )
        analyzers.append(whale_analyzer)

        return analyzers

    # ------ Strategy layer ---------------------------------------------

    def _build_strategy_layer(
        self,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> StrategyEngine:
        """
        Build StrategyEngine and register all concrete strategy classes.

        Strategies receive StrategyContext only; they do NOT call analytics
        or risk/execution directly.
        StrategyEngine → SignalProcessor → signal.generated (via EventBus).
        """
        strategy_config = StrategyConfig()

        engine = StrategyEngine(
            config=strategy_config,
            event_bus=event_bus,
            scheduler=scheduler,
        )

        # All concrete strategies to register
        _strategy_classes = [
            # Funding
            FundingDivergenceStrategy,
            FundingExtremeReversalStrategy,
            # Open Interest
            OIAnomalyStrategy,
            OIBreakoutConfirmationStrategy,
            OICapitulationStrategy,
            OIDivergenceStrategy,
            # Orderflow
            CvdDivergenceStrategy,
            OrderflowContinuationStrategy,
            OrderflowReversalStrategy,
            # Liquidations
            LiquidationCascadeStrategy,
            SqueezeReversalStrategy,
            # Liquidity
            EqualHighLowStrategy,
            LiquidityMapBiasStrategy,
            LiquiditySweepStrategy,
            StopHuntReversalStrategy,
            # Price Action
            FVGReactionStrategy,
            MarketStructureStrategy,
            SupportResistanceReactionStrategy,
            TrendContinuationStrategy,
            # Spoofing
            CompositeSpoofingStrategy,
            FakeLiquidityTrapStrategy,
            LayeringTrapStrategy,
            OrderPullReversalStrategy,
            PressureBluffReversalStrategy,
            SpoofingAbsorptionReversalStrategy,
            SpoofingReversalStrategy,
            # Spreads
            CrossExchangeArbStrategy,
            FundingAdjustedBasisStrategy,
            SpotFuturesBasisStrategy,
            SpreadMeanReversionStrategy,
            SpreadMomentumStrategy,
            # Whales
            WhaleAbsorptionStrategy,
            WhaleAccumulationStrategy,
            WhaleBreakoutStrategy,
            WhaleDistributionStrategy,
            WhaleLiquidationReversalStrategy,
            # Hybrid
            ConfluenceStrategy,
            LiquidationWhaleStrategy,
            LiquidityOrderflowReversalStrategy,
            MeanReversionStackStrategy,
            OIFundingSqueezeStrategy,
            TrendStackStrategy,
            WhaleOrderflowBreakoutStrategy,
        ]

        for strategy_cls in _strategy_classes:
            try:
                instance = strategy_cls(
                    config=strategy_config,
                    event_bus=event_bus,
                    scheduler=scheduler,
                )
                engine.registry.register_strategy(instance)
            except Exception as exc:
                self._logger.warning(
                    "Strategy registration skipped | strategy=%s error=%s",
                    strategy_cls.__name__,
                    exc,
                )

        return engine

    # ------ Risk layer -------------------------------------------------

    def _build_risk_layer(
        self,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> RiskManager:
        """
        RiskManager:
        - subscribes to signal.generated
        - publishes signal.confirmed / risk.position_*
        """
        risk_config = RiskConfig()
        return RiskManager(
            config=risk_config,
            event_bus=event_bus,
            scheduler=scheduler,
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

    # ------ Execution layer --------------------------------------------

    def _build_execution_layer(
        self,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        exchange_clients: dict[str, Any],
    ) -> tuple[OrderManager, PositionManager, SLTPManager, SmartExecution, TradeExecutor]:
        """
        Execution pipeline:
        signal.confirmed → TradeExecutor → OrderManager → exchange REST API
        TradeExecutor also listens to risk.position_close_requested / risk.kill_switch
        """
        rest_client = exchange_clients.get("binance_rest")
        rest_clients_map = {"binance": rest_client} if rest_client else {}

        order_manager = OrderManager(
            config=OrderManagerConfig(),
            event_bus=event_bus,
            scheduler=scheduler,
            exchange_clients=rest_clients_map,
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        position_manager = PositionManager(
            config=PositionManagerConfig(),
            event_bus=event_bus,
            scheduler=scheduler,
            exchange_clients=rest_clients_map,
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        sltp_manager = SLTPManager(
            config=SLTPManagerConfig(),
            order_manager=order_manager,
            event_bus=event_bus,
            scheduler=scheduler,
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        smart_execution = SmartExecution(
            config=SmartExecutionConfig(),
        )

        trade_executor = TradeExecutor(
            config=TradeExecutorConfig(),
            order_manager=order_manager,
            position_manager=position_manager,
            sltp_manager=sltp_manager,
            smart_execution=smart_execution,
            event_bus=event_bus,
            scheduler=scheduler,
            auto_subscribe=True,
            register_scheduler_jobs=True,
        )

        return order_manager, position_manager, sltp_manager, smart_execution, trade_executor

    # ------ News AI layer (parallel, non-trading) ----------------------

    def _build_news_layer(
        self,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> NewsAIService | None:
        if not self._enable_news:
            return None
        news_config = build_default_news_ai_config()
        return NewsAIService(
            event_bus=event_bus,
            scheduler=scheduler,
            config=news_config,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._logger.info(
            "TradingSystem starting | symbols=%s exchange=%s timeframe=%s news=%s",
            self._symbols,
            self._exchange,
            self._timeframe,
            self._enable_news,
        )

        # ---- Core infrastructure ----
        config = self._build_config()
        self._config = config

        event_bus = self._build_event_bus(config)
        self._event_bus = event_bus
        await event_bus.start()

        scheduler = self._build_scheduler(config, event_bus)
        self._scheduler = scheduler
        await scheduler.start()

        # ---- Data layer ----
        (
            candles_cache,
            orderbook_cache,
            trades_cache,
            funding_cache,
            oi_cache,
        ) = self._build_data_layer(config, event_bus, scheduler)

        # Register caches (subscribe to market.* topics)
        for cache in (candles_cache, orderbook_cache, trades_cache, funding_cache, oi_cache):
            cache.register()

        # ---- Exchange adapters ----
        exchange_clients = self._build_exchange_layer(config, event_bus, scheduler)

        # ---- Market stream orchestrator ----
        market_stream = MarketStream(
            config=config,
            event_bus=event_bus,
            exchange_clients=exchange_clients,
            scheduler=scheduler,
            stream_config=MarketStreamConfig(
                healthcheck_interval_seconds=30.0,
                start_clients_on_start=True,
                register_caches_on_start=False,  # registered above
            ),
            caches=[candles_cache, orderbook_cache, trades_cache, funding_cache, oi_cache],
        )

        # ---- Analytics layer ----
        analytics_services = self._build_analytics_layer(
            config, event_bus, scheduler, trades_cache, orderbook_cache
        )
        for svc in analytics_services:
            _try_register(svc)

        # ---- Strategy layer ----
        strategy_engine = self._build_strategy_layer(config, event_bus, scheduler)
        strategy_engine.register()

        # ---- Risk layer ----
        risk_manager = self._build_risk_layer(config, event_bus, scheduler)

        # ---- Execution layer ----
        (
            order_manager,
            position_manager,
            sltp_manager,
            smart_execution,
            trade_executor,
        ) = self._build_execution_layer(config, event_bus, scheduler, exchange_clients)

        # ---- News AI layer ----
        news_service = self._build_news_layer(event_bus, scheduler)

        # ---- Start all services in dependency order ----
        # Market stream (starts exchange WS clients + cache start hooks)
        await market_stream.start()

        # Analytics
        for svc in analytics_services:
            await _try_start(svc)

        # Strategy
        await strategy_engine.start()

        # Risk
        await risk_manager.start()

        # Execution
        await order_manager.start()
        await position_manager.start()
        await sltp_manager.start()
        await trade_executor.start()

        # News (parallel, non-blocking to trading pipeline)
        if news_service is not None:
            news_service.register()
            # NewsAIService uses Scheduler internally; no separate start() needed

        self._running = True

        # Store references for shutdown
        self._services = [
            trade_executor,
            sltp_manager,
            position_manager,
            order_manager,
            risk_manager,
            strategy_engine,
            *analytics_services,
            market_stream,
            scheduler,
            event_bus,
        ]

        self._logger.info(
            "TradingSystem started | services=%s strategies=%s",
            len(self._services),
            len(strategy_engine.registry.list_names()),
        )

        await event_bus.emit(
            "system.started",
            {
                "exchange": self._exchange,
                "symbols": self._symbols,
                "timeframe": self._timeframe,
                "started_at": time.time(),
            },
            source="trading_system",
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._logger.info("TradingSystem stopping...")

        if self._event_bus:
            await self._event_bus.emit(
                "system.stopping",
                {"stopped_at": time.time()},
                source="trading_system",
            )

        # Stop in reverse dependency order
        for svc in self._services:
            await _try_stop(svc)

        self._logger.info("TradingSystem stopped")
        self._shutdown_event.set()

    async def run_until_shutdown(self) -> None:
        """Block until SIGINT/SIGTERM or internal error."""
        loop = asyncio.get_running_loop()

        def _signal_handler(sig: signal.Signals) -> None:
            self._logger.info("Signal received | signal=%s", sig.name)
            loop.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler, sig)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main-thread fallback
                signal.signal(sig, lambda s, f: loop.create_task(self.stop()))

        await self._shutdown_event.wait()


# ============================================================================
# Helpers
# ============================================================================

async def _try_start(svc: Any) -> None:
    start = getattr(svc, "start", None)
    if callable(start):
        try:
            result = start()
            if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger = get_logger(__name__)
            logger.exception(
                "Service start failed | service=%s error=%s",
                svc.__class__.__name__,
                exc,
            )


async def _try_stop(svc: Any) -> None:
    stop = getattr(svc, "stop", None)
    if callable(stop):
        try:
            result = stop()
            if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger = get_logger(__name__)
            logger.exception(
                "Service stop failed | service=%s error=%s",
                svc.__class__.__name__,
                exc,
            )


def _try_register(svc: Any) -> None:
    register = getattr(svc, "register", None)
    if callable(register):
        try:
            register()
        except Exception as exc:
            logger = get_logger(__name__)
            logger.exception(
                "Service register failed | service=%s error=%s",
                svc.__class__.__name__,
                exc,
            )


# ============================================================================
# Entry point
# ============================================================================

async def _main() -> None:
    args = _parse_args()

    # Bootstrap logger before anything else
    init_logger(
        service_name="trading_system",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        json_logs=os.getenv("LOG_JSON", "false").lower() == "true",
        enable_file_logging=os.getenv("LOG_TO_FILE", "false").lower() == "true",
    )

    logger = get_logger(__name__, event_type="main")
    logger.info(
        "Starting production futures trading system | symbols=%s exchange=%s timeframe=%s",
        args.symbols,
        args.exchange,
        args.timeframe,
    )

    system = TradingSystem(
        symbols=args.symbols,
        exchange=args.exchange,
        timeframe=args.timeframe,
        enable_news=args.news,
        env_file=args.env,
    )

    try:
        await system.start()
        await system.run_until_shutdown()
    except Exception as exc:
        logger.exception("Fatal error in TradingSystem | error=%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())