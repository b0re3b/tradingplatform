#!/usr/bin/env python3
# trading_system/bootstrap.py

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Core
from core.config import Config
from core.event_bus import EventBus, EventPriority
from core.scheduler import Scheduler
from core.logger import init_logger, get_logger

# Data caches
from data.candles_cache import CandlesCache
from data.trades_cache import TradesCache
from data.orderbook_cache import OrderbookCache
from data.funding_cache import FundingCache
from data.open_interest_cache import OpenInterestCache

# Exchange adapters (Binance as execution + market data)
from exchanges.binance.binance_rest import BinanceRestClient, BinanceFuturesRestClientConfig
from exchanges.binance.binance_ws import BinanceWebSocketClient, BinanceWebSocketClientConfig

# Analytics
from analytics.orderflow.analyzer import OrderFlowAnalyzer
from analytics.orderflow.config import OrderFlowConfig
from analytics.open_interest.oi_analyzer import OIAnalyzer, OIAnalyzerConfig
from analytics.funding.funding_analyzer import FundingAnalyzer, FundingAnalyzerConfig
from analytics.spoofing.analyzer import SpoofingAnalyzer
from analytics.spoofing.config import SpoofingConfig
from analytics.price_action.price_action.analyzer import PriceActionAnalyzer, PriceActionAnalyzerConfig
from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.config import WhalesConfig
from analytics.liquidity.liquidity_service import LiquidityService
from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidations.liquidation_stream import LiquidationStream, LiquidationStreamConfig

# Strategy
from strategy.engine import StrategyEngine
from strategy.config import StrategyConfig
from strategy.registry import StrategyRegistry
from strategy.presets import build_default_strategy_config, build_default_strategy_registry
from strategy.strategies import *  # lazy imports, but we need factories
from strategy.strategies.orderflow import CvdDivergenceStrategy, OrderflowContinuationStrategy, OrderflowReversalStrategy
from strategy.strategies.price_action import MarketStructureStrategy, FVGReactionStrategy, SupportResistanceReactionStrategy, TrendContinuationStrategy
from strategy.strategies.open_interest import OIDivergenceStrategy, OIBreakoutConfirmationStrategy, OIAnomalyStrategy, OICapitulationStrategy
from strategy.strategies.funding import FundingDivergenceStrategy, FundingExtremeReversalStrategy
from strategy.strategies.liquidity import LiquiditySweepStrategy, StopHuntReversalStrategy, EqualHighLowStrategy, LiquidityMapBiasStrategy
from strategy.strategies.liquidations import LiquidationCascadeStrategy, SqueezeReversalStrategy
from strategy.strategies.spoofing import CompositeSpoofingStrategy, SpoofingReversalStrategy, FakeLiquidityTrapStrategy, OrderPullReversalStrategy, PressureBluffReversalStrategy, LayeringTrapStrategy, SpoofingAbsorptionReversalStrategy
from strategy.strategies.spreads import SpotFuturesBasisStrategy, CrossExchangeArbStrategy, FundingAdjustedBasisStrategy, SpreadMeanReversionStrategy, SpreadMomentumStrategy
from strategy.strategies.whales import WhaleAbsorptionStrategy, WhaleBreakoutStrategy, WhaleAccumulationStrategy, WhaleDistributionStrategy, WhaleLiquidationReversalStrategy
from strategy.strategies.hybrid import ConfluenceStrategy, LiquidationWhaleStrategy, LiquidityOrderflowReversalStrategy, MeanReversionStackStrategy, OIFundingSqueezeStrategy, TrendStackStrategy, WhaleOrderflowBreakoutStrategy

# Risk
from risk.risk_manager import RiskManager
from risk.config import RiskConfig

# Execution
from execution.trade_executor import TradeExecutor
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from execution.sl_tp_manager import SLTPManager
from execution.config import ExecutionConfig

# ----------------------------------------------------------------------
# 1. Logger initialization
# ----------------------------------------------------------------------
init_logger(
    service_name="trading_system",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("LOG_JSON", "false").lower() == "true",
    enable_file_logging=os.getenv("LOG_TO_FILE", "false").lower() == "true",
    log_dir=os.getenv("LOG_DIR", "logs"),
)
logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 2. Config (simplified – in production load from file)
# ----------------------------------------------------------------------
# For demo we create a minimal config; in real system you'd load from YAML/JSON.
class DummyConfig:
    pass

config = DummyConfig()
config.exchange = DummyConfig()
config.exchange.credentials = DummyConfig()
config.exchange.credentials.api_key = os.getenv("BINANCE_API_KEY", "")
config.exchange.credentials.api_secret = os.getenv("BINANCE_API_SECRET", "")
config.exchange.credentials.passphrase = ""  # not used for Binance
config.exchange.credentials.testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
config.exchange.rest_url = "https://fapi.binance.com"
config.exchange.ws_url = "wss://fstream.binance.com/ws"
config.exchange.timeout_seconds = 10.0
config.exchange.reconnect_delay = 5.0
config.exchange.max_reconnect_attempts = 20

# ----------------------------------------------------------------------
# 3. EventBus and Scheduler
# ----------------------------------------------------------------------
event_bus = EventBus(max_queue_size=20000, worker_count=6)
scheduler = Scheduler(event_bus=event_bus, tick_interval=0.2)

# ----------------------------------------------------------------------
# 4. Data Caches (subscribe to raw market events)
# ----------------------------------------------------------------------
candles_cache = CandlesCache(
    config=config,
    event_bus=event_bus,
    scheduler=scheduler,
    max_candles_per_key=2000,
)
trades_cache = TradesCache(
    config=config,
    event_bus=event_bus,
    scheduler=scheduler,
    max_trades_per_book=5000,
)
orderbook_cache = OrderbookCache(
    config=config,
    event_bus=event_bus,
    scheduler=scheduler,
    max_depth_per_side=200,
)
funding_cache = FundingCache(
    config=config,
    event_bus=event_bus,
    scheduler=scheduler,
)
open_interest_cache = OpenInterestCache(
    config=config,
    event_bus=event_bus,
    scheduler=scheduler,
)

# ----------------------------------------------------------------------
# 5. Exchange adapters (public market data + private execution)
# ----------------------------------------------------------------------
# Binance WebSocket for market data (public streams)
binance_ws = BinanceWebSocketClient(
    config=config,
    event_bus=event_bus,
    scheduler=scheduler,
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    streams=["trade", "depth", "kline"],
    depth_level="20",
    kline_interval="1m",
    enable_private_stream=True,   # for user data stream
)

# Binance REST for order execution (trading endpoints)
binance_rest = BinanceRestClient(
    config=config,
    event_bus=event_bus,
    use_demo=config.exchange.credentials.testnet,
)

# Optional: other exchanges for market data (Bybit, MEXC, OKX) can be added here
# but for simplicity we only use Binance.

# ----------------------------------------------------------------------
# 6. Analytics components
# ----------------------------------------------------------------------
# OrderFlow
orderflow_config = OrderFlowConfig()
orderflow_analyzer = OrderFlowAnalyzer(
    event_bus=event_bus,
    trades_cache=trades_cache,
    orderbook_cache=orderbook_cache,
    config=orderflow_config,
    scheduler=scheduler,
)

# Open Interest
oi_config = OIAnalyzerConfig(
    enabled=True,
    candle_topics=("market.candle.closed",),
    open_interest_topics=("market.open_interest.updated",),
)
oi_analyzer = OIAnalyzer(
    event_bus=event_bus,
    scheduler=scheduler,
    config=oi_config,
)

# Funding
funding_config = FundingAnalyzerConfig(enabled=True)
funding_analyzer = FundingAnalyzer(
    event_bus=event_bus,
    scheduler=scheduler,
    config=funding_config,
)

# Spoofing (needs orderbook_cache)
spoofing_config = SpoofingConfig(enabled=True, wall_detection={"max_levels_to_scan": 50})
spoofing_analyzer = SpoofingAnalyzer(
    event_bus=event_bus,
    scheduler=scheduler,
    config=spoofing_config,
    orderbook_cache=orderbook_cache,
)

# Price Action
price_action_config = PriceActionAnalyzerConfig(
    enable_market_structure=True,
    enable_support_resistance=True,
    enable_fair_value_gap=True,
    enable_liquidity_levels=True,
    enable_trend=True,
)
price_action_analyzer = PriceActionAnalyzer(
    symbol="BTCUSDT",  # Will be overridden per event, but required for constructor
    event_bus=event_bus,
    scheduler=scheduler,
    config=price_action_config,
    exchange="binance",
    market_type="usdm_futures",
)

# Whales (needs trades_cache, etc.)
whales_config = WhalesConfig(enabled=True)
whale_analyzer = WhaleAnalyzer(
    config=whales_config,
    event_bus=event_bus,
    scheduler=scheduler,
)

# Liquidity (needs orderbook_cache and CandlesCache)
liquidity_map = LiquidityMap()
liquidity_config = LiquidityConfig(enabled=True)
liquidity_service = LiquidityService(
    event_bus=event_bus,
    scheduler=scheduler,
    config=liquidity_config,
    liquidity_map=liquidity_map,
)

# Liquidation Stream
liquidation_config = LiquidationStreamConfig(
    input_topics=("market.liquidation",),
    enabled=True,
)
liquidation_stream = LiquidationStream(
    event_bus=event_bus,
    config=liquidation_config,
    scheduler=scheduler,
)

# ----------------------------------------------------------------------
# 7. Strategy Engine
# ----------------------------------------------------------------------
# Build strategy config (default preset)
strategy_config = build_default_strategy_config(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    preset_name="intraday",   # or "default", "scalping", "swing", etc.
    use_required_features=True,
)

# Factory map for concrete strategies
strategy_factories: Dict[str, Any] = {
    # Orderflow
    "cvd_divergence": CvdDivergenceStrategy,
    "orderflow_continuation": OrderflowContinuationStrategy,
    "orderflow_reversal": OrderflowReversalStrategy,
    # Price action
    "market_structure": MarketStructureStrategy,
    "fvg_reaction": FVGReactionStrategy,
    "support_resistance_reaction": SupportResistanceReactionStrategy,
    "trend_continuation": TrendContinuationStrategy,
    # Open interest
    "oi_divergence": OIDivergenceStrategy,
    "oi_breakout_confirmation": OIBreakoutConfirmationStrategy,
    "oi_anomaly": OIAnomalyStrategy,
    "oi_capitulation": OICapitulationStrategy,
    # Funding
    "funding_divergence": FundingDivergenceStrategy,
    "funding_extreme_reversal": FundingExtremeReversalStrategy,
    # Liquidity
    "liquidity_sweep": LiquiditySweepStrategy,
    "stop_hunt_reversal": StopHuntReversalStrategy,
    "equal_high_low": EqualHighLowStrategy,
    "liquidity_map_bias": LiquidityMapBiasStrategy,
    # Liquidations
    "liquidation_cascade": LiquidationCascadeStrategy,
    "squeeze_reversal": SqueezeReversalStrategy,
    # Spoofing
    "composite_spoofing": CompositeSpoofingStrategy,
    "spoofing_reversal": SpoofingReversalStrategy,
    "fake_liquidity_trap": FakeLiquidityTrapStrategy,
    "order_pull_reversal": OrderPullReversalStrategy,
    "pressure_bluff_reversal": PressureBluffReversalStrategy,
    "layering_trap": LayeringTrapStrategy,
    "spoofing_absorption_reversal": SpoofingAbsorptionReversalStrategy,
    # Spreads
    "spot_futures_basis": SpotFuturesBasisStrategy,
    "cross_exchange_arb": CrossExchangeArbStrategy,
    "funding_adjusted_basis": FundingAdjustedBasisStrategy,
    "spread_mean_reversion": SpreadMeanReversionStrategy,
    "spread_momentum": SpreadMomentumStrategy,
    # Whales
    "whale_absorption": WhaleAbsorptionStrategy,
    "whale_breakout": WhaleBreakoutStrategy,
    "whale_accumulation": WhaleAccumulationStrategy,
    "whale_distribution": WhaleDistributionStrategy,
    "whale_liquidation_reversal": WhaleLiquidationReversalStrategy,
    # Hybrid
    "confluence": ConfluenceStrategy,
    "liquidation_whale": LiquidationWhaleStrategy,
    "liquidity_orderflow_reversal": LiquidityOrderflowReversalStrategy,
    "mean_reversion_stack": MeanReversionStackStrategy,
    "oi_funding_squeeze": OIFundingSqueezeStrategy,
    "trend_stack": TrendStackStrategy,
    "whale_orderflow_breakout": WhaleOrderflowBreakoutStrategy,
}

registry = build_default_strategy_registry(
    config=strategy_config,
    event_bus=event_bus,
    scheduler=scheduler,
    strategy_factories=strategy_factories,
    replace=False,
    emit_events=True,
)
strategy_engine = StrategyEngine(
    config=strategy_config,
    event_bus=event_bus,
    scheduler=scheduler,
    registry=registry,
)

# ----------------------------------------------------------------------
# 8. Risk Manager
# ----------------------------------------------------------------------
risk_config = RiskConfig()  # default configuration
risk_manager = RiskManager(
    config=risk_config,
    event_bus=event_bus,
    scheduler=scheduler,
    auto_subscribe=True,
    register_scheduler_jobs=False,  # we start manually
)

# ----------------------------------------------------------------------
# 9. Execution Layer
# ----------------------------------------------------------------------
execution_config = ExecutionConfig()
order_manager = OrderManager(
    config=execution_config,
    event_bus=event_bus,
    exchange_client=binance_rest,  # BinanceRestClient implements required protocols
)
position_manager = PositionManager(
    config=execution_config,
    event_bus=event_bus,
    exchange_client=binance_rest,
)
sltp_manager = SLTPManager(
    config=execution_config,
    event_bus=event_bus,
    order_manager=order_manager,
)
trade_executor = TradeExecutor(
    config=execution_config,
    event_bus=event_bus,
    order_manager=order_manager,
    position_manager=position_manager,
    sltp_manager=sltp_manager,
    market_context_provider=None,  # optional
)

# ----------------------------------------------------------------------
# 10. (Optional) MarketStream orchestrator – if you want centralized start/stop
# ----------------------------------------------------------------------
# We can manually start components in order; MarketStream not strictly needed.

# ----------------------------------------------------------------------
# 11. Lifecycle: register, start, and keep running
# ----------------------------------------------------------------------
async def start_system():
    """Start all components in dependency order."""
    # First start EventBus and Scheduler
    await event_bus.start()
    await scheduler.start()

    # Data caches (register only, they need EventBus subscriptions)
    candles_cache.register()
    trades_cache.register()
    orderbook_cache.register()
    funding_cache.register()
    open_interest_cache.register()

    # Exchange adapters
    await binance_ws.start()
    await binance_rest.start()

    # Analytics (register and start)
    # For components that have both register() and start(), we call register first.
    for comp in [
        orderflow_analyzer,
        oi_analyzer,
        funding_analyzer,
        spoofing_analyzer,
        price_action_analyzer,
        whale_analyzer,
        liquidity_service,
        liquidation_stream,
    ]:
        if hasattr(comp, "register"):
            comp.register()
        if hasattr(comp, "start"):
            await comp.start()

    # Strategy engine (register + start)
    strategy_engine.register()
    await strategy_engine.start()

    # Risk manager
    risk_manager.register()
    await risk_manager.start()

    # Execution components (register + start)
    order_manager.register()
    await order_manager.start()
    position_manager.register()
    await position_manager.start()
    sltp_manager.register()
    await sltp_manager.start()
    trade_executor.register()
    await trade_executor.start()

    logger.info("All components started successfully")
    # Keep running until interrupted
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

async def stop_system():
    """Graceful shutdown in reverse order."""
    # Execution
    for comp in [trade_executor, sltp_manager, position_manager, order_manager]:
        if hasattr(comp, "stop"):
            await comp.stop()
    # Risk
    await risk_manager.stop()
    # Strategy
    await strategy_engine.stop()
    # Analytics (stop in reverse order)
    for comp in [
        liquidation_stream,
        liquidity_service,
        whale_analyzer,
        price_action_analyzer,
        spoofing_analyzer,
        funding_analyzer,
        oi_analyzer,
        orderflow_analyzer,
    ]:
        if hasattr(comp, "stop"):
            await comp.stop()
    # Exchange adapters
    await binance_ws.stop()
    await binance_rest.stop()
    # Caches (no stop needed usually, but they may have cleanup jobs)
    # Scheduler and EventBus
    await scheduler.stop()
    await event_bus.stop()
    logger.info("System stopped")

async def main():
    try:
        await start_system()
    finally:
        await stop_system()

if __name__ == "__main__":
    asyncio.run(main())